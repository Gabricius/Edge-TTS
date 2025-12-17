import io
import asyncio
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge-TTS Microservice", version="1.6.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

# --- ROTA HOME ---
@app.get("/")
def home():
    return {"status": "online", "message": "Serviço Ativo. Use POST em /generate"}

# --- ROTA DE VOZES ---
@app.get("/voices")
async def list_voices():
    try:
        voices = await edge_tts.list_voices()
        curated = [
            {"Name": v["ShortName"], "Gender": v["Gender"], "Locale": v["Locale"]}
            for v in voices if "Neural" in v["ShortName"]
        ]
        return {"count": len(curated), "voices": curated}
    except Exception as e:
        return {"error": str(e)}

# --- ROTA DE GERAÇÃO ---
@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Novo Pedido ---")
    logger.info(f"Voz: {request.voice}")
    
    try:
        # Monta os argumentos dinamicamente para evitar enviar lixo
        communicate_params = {
            "text": request.text,
            "voice": request.voice
        }
        
        # Só adiciona rate/pitch se forem DIFERENTES do padrão
        if request.rate and request.rate not in ["+0%", "-0%"]:
             communicate_params["rate"] = request.rate
        if request.pitch and request.pitch not in ["+0Hz", "-0Hz"]:
             communicate_params["pitch"] = request.pitch

        # Desempacota os argumentos (**communicate_params)
        communicate = edge_tts.Communicate(**communicate_params)
        
        audio_buffer = io.BytesIO()
        submaker = edge_tts.SubMaker()
        message_counts = {"audio": 0, "WordBoundary": 0}

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
                message_counts["audio"] += 1
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
                message_counts["WordBoundary"] += 1

        logger.info(f"Stream OK. Chunks: {message_counts['audio']} | Legendas: {message_counts['WordBoundary']}")

        # 1. Processa Áudio
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # 2. Processa Legenda
        srt_content = ""
        if message_counts["WordBoundary"] > 0:
            try:
                srt_content = submaker.generate_subs()
                srt_content = srt_content.replace("WEBVTT\n\n", "")
            except Exception as e:
                logger.error(f"Erro legenda: {e}")
                srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{request.text}"
        else:
            srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{request.text}"

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
        logger.error(f"Erro Fatal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
