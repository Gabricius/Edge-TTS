import io
import asyncio
import base64
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

app = FastAPI(title="Edge-TTS Microservice", version="1.0.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def vtt_to_srt_buffer(vtt_content: str) -> str:
    lines = vtt_content.strip().splitlines()
    srt_output = []
    counter = 1
    timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2})\.(\d{3})\s-->\s(\d{2}:\d{2}:\d{2})\.(\d{3})')
    
    for line in lines:
        if line.startswith("WEBVTT") or line.startswith("X-TIMESTAMP") or line.strip() == "":
            continue
        match = timestamp_pattern.search(line)
        if match:
            new_timestamp = f"{match.group(1)},{match.group(2)} --> {match.group(3)},{match.group(4)}"
            srt_output.append(str(counter))
            srt_output.append(new_timestamp)
            counter += 1
        else:
            srt_output.append(line)
            srt_output.append("")
    return "\n".join(srt_output)

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    try:
        communicate = edge_tts.Communicate(text=request.text, voice=request.voice, rate=request.rate, pitch=request.pitch)
        audio_buffer = io.BytesIO()
        submaker = edge_tts.SubMaker()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
                
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        vtt_content = submaker.generate_subs()
        srt_content = vtt_to_srt_buffer(vtt_content)
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