import io
import asyncio
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge-TTS Factory", version="7.0.0 (Audio Only)")

class TTSRequest(BaseModel):
    text: str
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"Gerando áudio para: {len(request.text)} chars")
    try:
        # Garante parâmetros limpos
        rate = request.rate if request.rate else "+0%"
        pitch = request.pitch if request.pitch else "+0Hz"
        
        communicate = edge_tts.Communicate(request.text, request.voice, rate=rate, pitch=pitch)
        audio_buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
                
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return {
            "status": "success",
            "data": { "audio_base64": audio_base64 }
        }
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
