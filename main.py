import io
import asyncio
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

app = FastAPI(title="Edge-TTS Microservice", version="1.1.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def format_srt_time(ticks):
    """Converte ticks (100ns) para formato SRT (HH:MM:SS,mmm)"""
    seconds = ticks / 10_000_000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def events_to_srt(events) -> str:
    """Gera legenda SRT diretamente dos eventos de WordBoundary"""
    srt_output = []
    counter = 1
    
    # Agrupar palavras em frases pode ser complexo, aqui faremos palavra/trecho por trecho
    # O Edge-TTS envia "WordBoundaries" que são basicamente as legendas prontas
    for event in events:
        start_time = format_srt_time(event['offset'])
        end_time = format_srt_time(event['offset'] + event['duration'])
        text = event['text']
        
        # Formato SRT
        srt_output.append(str(counter))
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(text)
        srt_output.append("") # Linha em branco obrigatória
        counter += 1
        
    return "\n".join(srt_output)

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    try:
        communicate = edge_tts.Communicate(text=request.text, voice=request.voice, rate=request.rate, pitch=request.pitch)
        audio_buffer = io.BytesIO()
        word_events = [] # Lista para guardar os eventos de legenda manualmente
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # Guardamos o evento na nossa lista em vez de usar SubMaker
                word_events.append(chunk)
                
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # Gera o SRT diretamente da nossa lista de eventos
        srt_content = events_to_srt(word_events)
        srt_base64 = base64.b64encode(srt_content.encode('utf-8')).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "audio_base64": audio_base64,
                "srt_base64": srt_base64,
                "metadata": {"voice": request.voice}
            }
        }
    except Exception as e:
        print(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
