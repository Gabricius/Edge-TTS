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

app = FastAPI(title="Edge-TTS Microservice", version="1.4.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def clean_param(param):
    """Remove parâmetros neutros que podem causar erro na API"""
    if param in ["+0%", "-0%", "+0Hz", "-0Hz"]:
        return None
    return param

# --- ROTA DA PÁGINA INICIAL (Resolve o erro "Not Found" no navegador) ---
@app.get("/")
def home():
    return {"status": "online", "message": "O serviço Edge-TTS está rodando! Use o endpoint /generate (POST) no n8n."}

# --- ROTA PARA LISTAR VOZES (Para você consultar no navegador) ---
@app.get("/voices")
async def list_voices():
    try:
        voices = await edge_tts.list_voices()
        # Filtra para mostrar apenas as vozes neurais principais
        curated = [
            {"Name": v["ShortName"], "Gender": v["Gender"], "Locale": v["Locale"]}
            for v in voices if "Neural" in v["ShortName"]
        ]
        return {"count": len(curated), "voices": curated}
    except Exception as e:
        return {"error": str(e)}

# --- ROTA PRINCIPAL DE GERAÇÃO (Usada pelo n8n) ---
@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Novo Pedido ---")
    logger.info(f"Texto (inicio): {request.text[:50]}...")
    
    try:
        final_rate = clean_param(request.rate)
        final_pitch = clean_param(request.pitch)

        communicate = edge_tts.Communicate(
            text=request.text, 
            voice=request.voice, 
            rate=final_rate, 
            pitch=final_pitch
        )
        
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

        logger.info(f"Stream finalizado. Audio chunks: {message_counts['audio']}, Legenda events: {message_counts['WordBoundary']}")

        # Prepara Audio
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # Prepara Legenda
        srt_content = ""
        if message_counts["WordBoundary"] > 0:
            try:
                srt_content = submaker.generate_subs()
                # Remove cabeçalho WEBVTT se existir e ajusta formato se necessário
                # O edge-tts mais novo já gera VTT limpo, mas o n8n precisa de SRT
                # Converter VTT para SRT simples (trocar ponto por virgula no tempo)
                # Mas para simplificar, vamos enviar o que vier e tratar no n8n se precisar,
                # ou fazer um replace simples aqui:
                srt_content = srt_content.replace("WEBVTT\n\n", "")
            except Exception as e:
                logger.error(f"Erro ao gerar legenda: {e}")
                srt_content = "1\n00:00:00,000 --> 00:00:05,000\nErro na geração da legenda."
        else:
            # Fallback se a Microsoft não mandar tempos
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
