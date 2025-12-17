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

app = FastAPI(title="Edge-TTS Pro", version="3.1.0")

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
    Agrupa palavras individuais em frases de 4-6 palavras.
    """
    if not events:
        return []

    captions = []
    current_chunk = []
    
    for event in events:
        current_chunk.append(event)
        
        # Fecha o bloco se atingir limite ou pontuação
        is_end_sentence = event['text'].strip()[-1] in ['.', '?', '!', ':']
        if len(current_chunk) >= max_words or (len(current_chunk) >= 3 and is_end_sentence):
            start_time = current_chunk[0]['offset']
            end_time = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
            text = " ".join([e['text'] for e in current_chunk])
            
            captions.append({"start": start_time, "end": end_time, "text": text})
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
    """
    Gera áudio para um segmento.
    CORREÇÃO V3.1: Garante retorno mesmo se falhar a legenda.
    """
    last_audio = b""
    last_events = []

    for attempt in range(retries + 1):
        try:
            # Garante que rate/pitch sejam strings válidas
            safe_rate = rate if rate else "+0%"
            safe_pitch = pitch if pitch else "+0Hz"

            communicate = edge_tts.Communicate(text, voice, rate=safe_rate, pitch=safe_pitch)
            audio_buffer = io.BytesIO()
            events = []
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    events.append(chunk)
            
            # Salva o resultado desta tentativa
            last_audio = audio_buffer.getvalue()
            last_events = events

            # Lógica de Validação
            # Se temos áudio mas SEM eventos (legenda), e ainda temos tentativas...
            if len(last_audio) > 0 and not events:
                if attempt < retries:
                    logger.warning(f"Tentativa {attempt+1}: Áudio OK, mas sem legendas. Tentando de novo...")
                    await asyncio.sleep(1)
                    continue # Tenta de novo
                else:
                    # Se for a última tentativa, desiste e retorna o que tem
                    logger.warning("Última tentativa sem legendas. Retornando áudio bruto.")
                    return last_audio, []
            
            # Sucesso total
            return last_audio, last_events
            
        except Exception as e:
            logger.error(f"Erro na tentativa {attempt+1}: {e}")
            if attempt == retries:
                # Se falhar tudo (ex: erro de rede), retorna erro ou vazio
                raise e
            await asyncio.sleep(1)

    # Fallback de segurança (caso o loop termine de forma inesperada)
    return last_audio, last_events

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Novo Pedido V3.1 ({len(request.text)} chars) ---")
    
    # 1. DIVISÃO DO TEXTO
    # Divide em blocos menores (1000 chars) para facilitar a vida da Microsoft
    raw_segments = re.split(r'(?<=[.?!])\s+', request.text)
    segments = []
    current = ""
    for s in raw_segments:
        if len(current) + len(s) < 1000: # Reduzi para 1000 por segurança
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
            
            # Chama a função corrigida
            seg_audio_bytes, seg_events = await generate_segment(seg, request.voice, request.rate, request.pitch)
            
            # Cola o áudio
            full_audio.write(seg_audio_bytes)
            
            # Processa eventos (se houver)
            if seg_events:
                for event in seg_events:
                    event['offset'] += global_offset
                    all_events.append(event)
                
                last = seg_events[-1]
                global_offset = last['offset'] + last['duration'] + 2_000_000 # +200ms
            else:
                # FALLBACK INTELIGENTE
                # Se não vieram eventos, estimamos o tempo baseados no tamanho do texto
                # para que o próximo parágrafo não comece em cima deste.
                # Estimativa: 1 char ~= 80ms (800,000 ticks)
                logger.info(f"Segmento {i+1} sem legendas. Usando tempo estimado.")
                estimated_duration = len(seg) * 800_000
                
                # Criamos um 'evento falso' para que a legenda não fique vazia
                all_events.append({
                    "offset": global_offset,
                    "duration": estimated_duration,
                    "text": seg # Coloca o texto todo do bloco numa legenda só
                })
                global_offset += estimated_duration

        # 2. AGRUPAMENTO (4-6 Palavras)
        grouped_captions = group_words_to_captions(all_events, max_words=6)
        
        # 3. GERAÇÃO FINAL
        srt_content = generate_srt_string(grouped_captions)
        
        # Fallback Final
        if not srt_content:
            srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{request.text[:50]}..."

        return {
            "status": "success",
            "data": {
                "audio_base64": base64.b64encode(full_audio.getvalue()).decode('utf-8'),
                "srt_base64": base64.b64encode(srt_content.encode('utf-8')).decode('utf-8'),
                "metadata": {
                    "chunks": len(segments), 
                    "voice": request.voice
                }
            }
        }

    except Exception as e:
        logger.error(f"FATAL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
