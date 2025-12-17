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

app = FastAPI(title="Edge-TTS Pro", version="4.0.0 (Micro-Batch)")

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
    """Agrupa palavras em frases curtas (4-6 palavras)"""
    if not events: return []

    captions = []
    current_chunk = []
    
    for event in events:
        current_chunk.append(event)
        
        # Lógica de quebra: Pontuação ou limite de palavras
        text_content = event.get('text', '').strip()
        is_end_sentence = text_content and text_content[-1] in ['.', '?', '!', ':']
        
        if len(current_chunk) >= max_words or (len(current_chunk) >= 3 and is_end_sentence):
            start = current_chunk[0]['offset']
            # O fim é o offset da última palavra + duração dela
            end = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
            text = " ".join([e['text'] for e in current_chunk])
            
            captions.append({"start": start, "end": end, "text": text})
            current_chunk = []
    
    # Processa o que sobrou no buffer
    if current_chunk:
        start = current_chunk[0]['offset']
        end = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
        text = " ".join([e['text'] for e in current_chunk])
        captions.append({"start": start, "end": end, "text": text})
            
    return captions

def generate_srt_string(captions):
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
    """Gera áudio para um micro-segmento"""
    for attempt in range(retries + 1):
        try:
            # Garante strings limpas
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
            
            # Validação crítica: Se veio áudio mas sem legenda, é falha.
            if audio_buffer.getbuffer().nbytes > 0 and not events:
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue 
                else:
                    return audio_buffer.getvalue(), [] # Desiste e entrega sem legenda
            
            return audio_buffer.getvalue(), events
            
        except Exception as e:
            if attempt == retries: raise e
            await asyncio.sleep(1)

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Processando V4.0 ({len(request.text)} chars) ---")
    
    # 1. DIVISÃO MICRO-BATCH (Limite 250 chars)
    # Isso força o envio frase a frase, garantindo que a Microsoft mande os tempos.
    raw_sentences = re.split(r'(?<=[.?!])\s+', request.text)
    segments = []
    current_chunk = ""
    
    for s in raw_sentences:
        # Se a frase atual + a nova frase for menor que 250, agrupa
        if len(current_chunk) + len(s) < 250:
            current_chunk += s + " "
        else:
            if current_chunk: segments.append(current_chunk.strip())
            current_chunk = s + " "
    if current_chunk: segments.append(current_chunk.strip())

    full_audio = io.BytesIO()
    all_events = []
    global_offset = 0
    
    try:
        for i, seg in enumerate(segments):
            if not seg.strip(): continue
            
            # Gera o micro-segmento
            seg_audio, seg_events = await generate_segment(seg, request.voice, request.rate, request.pitch)
            
            full_audio.write(seg_audio)
            
            if seg_events:
                # Ajusta os tempos relativos para absolutos
                for event in seg_events:
                    event['offset'] += global_offset
                    all_events.append(event)
                
                # Atualiza o relógio global com precisão
                last = seg_events[-1]
                # Adiciona um micro-silêncio (50ms) para não atropelar
                global_offset = last['offset'] + last['duration'] + 500_000 
            else:
                # Fallback Ajustado (50ms por char em vez de 80ms)
                # Só entra aqui se a API da Microsoft falhar totalmente nesse trecho
                duration = len(seg) * 500_000 
                
                # Cria evento falso para não quebrar a legenda
                all_events.append({
                    "offset": global_offset,
                    "duration": duration,
                    "text": seg
                })
                global_offset += duration

            # Pausa mínima para evitar 429 Too Many Requests
            await asyncio.sleep(0.1)

        # 2. AGRUPAMENTO (Onde a mágica visual acontece)
        grouped_captions = group_words_to_captions(all_events, max_words=5)
        
        # 3. GERAÇÃO FINAL
        srt_content = generate_srt_string(grouped_captions)
        
        if not srt_content:
             srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{request.text[:50]}..."

        return {
            "status": "success",
            "data": {
                "audio_base64": base64.b64encode(full_audio.getvalue()).decode('utf-8'),
                "srt_base64": base64.b64encode(srt_content.encode('utf-8')).decode('utf-8'),
                "metadata": {"segments": len(segments), "voice": request.voice}
            }
        }

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
