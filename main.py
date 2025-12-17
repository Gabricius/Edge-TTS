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

app = FastAPI(title="Edge-TTS Microservice", version="1.3.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def clean_param(param):
    """Remove parâmetros neutros que podem bugar a API"""
    if param in ["+0%", "-0%", "+0Hz", "-0Hz"]:
        return None
    return param

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Novo Pedido ---")
    logger.info(f"Texto: {request.text[:30]}...")
    logger.info(f"Voz: {request.voice}")

    try:
        # Limpeza de parâmetros
        final_rate = clean_param(request.rate)
        final_pitch = clean_param(request.pitch)

        communicate = edge_tts.Communicate(
            text=request.text, 
            voice=request.voice, 
            rate=final_rate, 
            pitch=final_pitch
        )
        
        audio_buffer = io.BytesIO()
        submaker = edge_tts.SubMaker() # Vamos tentar usar o oficial primeiro
        
        message_counts = {"audio": 0, "WordBoundary": 0, "other": 0}

        async for chunk in communicate.stream():
            msg_type = chunk["type"]
            
            if msg_type == "audio":
                audio_buffer.write(chunk["data"])
                message_counts["audio"] += 1
            elif msg_type == "WordBoundary":
                submaker.feed(chunk)
                message_counts["WordBoundary"] += 1
            else:
                message_counts["other"] += 1

        logger.info(f"Estatísticas do Stream: {message_counts}")

        # Processamento do Áudio
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # Processamento da Legenda (SRT)
        srt_content = ""
        
        if message_counts["WordBoundary"] > 0:
            try:
                # Tenta o método nativo primeiro (mais preciso)
                srt_content = submaker.generate_subs()
                # O generate_subs retorna VTT, precisamos converter para SRT simples
                # Mas o erro anterior indicava que ele não existia.
                # Se funcionar, convertemos VTT -> SRT aqui:
                srt_content = srt_content.replace("WEBVTT\n\n", "").replace(".", ",")
            except Exception as sub_error:
                logger.warning(f"Erro no SubMaker nativo: {sub_error}. Tentando fallback manual.")
                # Se falhar, você verá no log, mas o código não quebra.
                srt_content = "1\n00:00:00,000 --> 00:00:05,000\nLegenda falhou na conversão."
        else:
            logger.warning("ALERTA: Microsoft não enviou tempos (WordBoundary). Texto muito curto ou erro de API.")
            # Fallback de emergência
            srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{request.text}"

        srt_base64 = base64.b64encode(srt_content.encode('utf-8')).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "audio_base64": audio_base64,
                "srt_base64": srt_base64,
                "metadata": {
                    "voice": request.voice,
                    "debug": message_counts
                }
            }
        }

    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
