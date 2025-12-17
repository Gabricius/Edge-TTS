import io
import asyncio
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

# Configuração de Logs (Para aparecer no Easypanel)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge-TTS Microservice", version="1.2.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def format_srt_time(ticks):
    seconds = ticks / 10_000_000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def events_to_srt(events) -> str:
    srt_output = []
    counter = 1
    for event in events:
        start_time = format_srt_time(event['offset'])
        end_time = format_srt_time(event['offset'] + event['duration'])
        text = event['text']
        srt_output.append(str(counter))
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(text)
        srt_output.append("") 
        counter += 1
    return "\n".join(srt_output)

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"Recebendo pedido: Voz={request.voice}, Texto='{request.text[:20]}...'")
    try:
        communicate = edge_tts.Communicate(text=request.text, voice=request.voice, rate=request.rate, pitch=request.pitch)
        audio_buffer = io.BytesIO()
        word_events = []
        
        # Processa o stream
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                word_events.append(chunk)
        
        logger.info(f"Stream finalizado. Tamanho audio: {audio_buffer.getbuffer().nbytes} bytes. Eventos legenda: {len(word_events)}")

        # Lógica de Falback: Se não veio legenda (texto curto), cria uma manual
        if len(word_events) == 0:
            logger.warning("Nenhum evento de legenda recebido. Criando legenda forçada.")
            # Cria uma legenda genérica de 0s a 2s
            word_events.append({
                "offset": 0,
                "duration": 20_000_000, # 2 segundos em ticks
                "text": request.text
            })

        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        srt_content = events_to_srt(word_events)
        srt_base64 = base64.b64encode(srt_content.encode('utf-8')).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "audio_base64": audio_base64,
                "srt_base64": srt_base64,
                "metadata": {
                    "voice": request.voice, 
                    "debug_events_count": len(word_events)
                }
            }
        }
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
