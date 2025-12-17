import io
import asyncio
import base64
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge-TTS Pro", version="3.0.0")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def format_srt_time(ticks):
    """Converte tempo Microsoft (ticks) para SRT"""
    seconds = ticks / 10_000_000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def group_words_to_captions(events, max_words=6):
    """
    Agrupa palavras individuais em frases de 4-6 palavras para legendas mais naturais.
    """
    captions = []
    current_chunk = []
    
    for event in events:
        current_chunk.append(event)
        
        # Se atingiu o limite de palavras ou encontrou pontuação final, fecha o bloco
        is_end_sentence = event['text'].strip()[-1] in ['.', '?', '!', ':']
        if len(current_chunk) >= max_words or (len(current_chunk) >= 3 and is_end_sentence):
            # Cria a legenda combinada
            start_time = current_chunk[0]['offset']
            # O fim é o offset da última palavra + sua duração
            end_time = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
            text = " ".join([e['text'] for e in current_chunk])
            
            captions.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
            current_chunk = []
    
    # Adiciona o que sobrou
    if current_chunk:
        start_time = current_chunk[0]['offset']
        end_time = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
        text = " ".join([e['text'] for e in current_chunk])
        captions.append({"start": start_time, "end": end_time, "text": text})
            
    return captions

def generate_srt_string(captions):
    """Gera o texto final do arquivo SRT"""
    output = []
    for i, cap in enumerate(captions, 1):
        start = format_srt_time(cap['start'])
        end = format_srt_time(cap['end'])
        output.append(str(i))
        output.append(f"{start} --> {end}")
        output.append(cap['text'])
        output.append("")
    return "\n".join(output)

async def generate_segment(text, voice, rate, pitch, retries=2):
    """Gera áudio para um segmento com tentativas automáticas em caso de erro"""
    for attempt in range(retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            audio = io.BytesIO()
            events = []
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    events.append(chunk)
            
            # Validação: Se veio áudio mas sem eventos, tenta de novo
            if audio.getbuffer().nbytes > 0 and not events:
                logger.warning(f"Tentativa {attempt+1}: Audio gerado sem legendas. Tentando novamente...")
                await asyncio.sleep(1)
                continue
                
            return audio.getvalue(), events
            
        except Exception as e:
            logger.error(f"Erro na tentativa {attempt+1}: {e}")
            if attempt == retries:
                raise e
            await asyncio.sleep(2) # Espera antes de tentar de novo

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Processando Texto ({len(request.text)} chars) ---")
    
    # 1. DIVISÃO SEGURA (Limite de 1500 chars para garantir WordBoundary)
    # Regex divide por quebra de linha ou pontuação se a linha for muito longa
    raw_segments = re.split(r'(?<=[.?!])\s+', request.text)
    segments = []
    current = ""
    for s in raw_segments:
        if len(current) + len(s) < 1500:
            current += s + " "
        else:
            segments.append(current.strip())
            current = s + " "
    if current: segments.append(current.strip())

    full_audio = io.BytesIO()
    all_events = []
    global_offset = 0
    
    try:
        for i, seg in enumerate(segments):
            if not seg.strip(): continue
            
            # Gera com retry automático
            seg_audio_bytes, seg_events = await generate_segment(seg, request.voice, request.rate, request.pitch)
            
            # Cola o áudio
            full_audio.write(seg_audio_bytes)
            
            # Ajusta os tempos dos eventos
            for event in seg_events:
                event['offset'] += global_offset
                all_events.append(event)
            
            # Atualiza o relógio global
            if seg_events:
                last = seg_events[-1]
                global_offset = last['offset'] + last['duration'] + 1_000_000 # +100ms de margem
            else:
                # Fallback de tempo se falhar totalmente
                global_offset += len(seg) * 500_000

        # 2. AGRUPAMENTO INTELIGENTE (4-6 Palavras)
        grouped_captions = group_words_to_captions(all_events, max_words=6)
        
        # 3. GERAÇÃO FINAL
        srt_content = generate_srt_string(grouped_captions)
        
        if not srt_content:
            # Fallback final se TUDO falhar
            srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{request.text[:100]}..."

        return {
            "status": "success",
            "data": {
                "audio_base64": base64.b64encode(full_audio.getvalue()).decode('utf-8'),
                "srt_base64": base64.b64encode(srt_content.encode('utf-8')).decode('utf-8'),
                "metadata": {"chunks": len(segments), "captions": len(grouped_captions)}
            }
        }

    except Exception as e:
        logger.error(f"FATAL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
