import io
import asyncio
import base64
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import edge_tts
import uvicorn

# --- CONFIGURAÇÃO DE CALIBRAÇÃO ---
# Se a legenda ficar curta demais, AUMENTE este número.
# Se ficar longa demais, DIMINUA.
# 50 = Rápido | 72 = Normal/Francisca | 85 = Vozes Lentas
MS_PER_CHAR_ESTIMATE = 72 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge-TTS Pro", version="6.0.0 (Calibrated)")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "pt-BR-FranciscaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

def format_srt_time(ticks):
    """Converte ticks (100ns) para formato SRT"""
    seconds = ticks / 10_000_000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def estimate_word_timings(text, start_offset, duration_ticks):
    """
    Distribui o tempo total do áudio proporcionalmente pelo número de letras de cada palavra.
    Isso é mais preciso que dividir igualmente por palavra.
    """
    words = text.split()
    if not words: return []
    
    total_chars = sum(len(w) for w in words)
    if total_chars == 0: total_chars = 1
    
    simulated_events = []
    current_offset = start_offset
    
    for word in words:
        # Calcula duração da palavra baseada no nº de letras dela
        word_weight = len(word) / total_chars
        word_duration = duration_ticks * word_weight
        
        simulated_events.append({
            "offset": current_offset,
            "duration": word_duration,
            "text": word
        })
        current_offset += word_duration
        
    return simulated_events

def group_words_to_captions(events, max_words=5):
    if not events: return []
    captions = []
    current_chunk = []
    
    for event in events:
        current_chunk.append(event)
        text_content = event.get('text', '').strip()
        is_end_sentence = text_content and text_content[-1] in ['.', '?', '!', ':']
        
        if len(current_chunk) >= max_words or (len(current_chunk) >= 2 and is_end_sentence):
            start = current_chunk[0]['offset']
            end = current_chunk[-1]['offset'] + current_chunk[-1]['duration']
            text = " ".join([e['text'] for e in current_chunk])
            captions.append({"start": start, "end": end, "text": text})
            current_chunk = []
    
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
    for attempt in range(retries + 1):
        try:
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
            
            if audio_buffer.getbuffer().nbytes > 0 and not events:
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue 
                else:
                    return audio_buffer.getvalue(), [] 
            
            return audio_buffer.getvalue(), events
            
        except Exception as e:
            if attempt == retries: raise e
            await asyncio.sleep(1)

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    logger.info(f"--- Processando V6.0 ({len(request.text)} chars) ---")
    
    # 1. DIVISÃO: 250 chars
    raw_sentences = re.split(r'(?<=[.?!])\s+', request.text)
    segments = []
    current_chunk = ""
    for s in raw_sentences:
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
            
            seg_audio, seg_events = await generate_segment(seg, request.voice, request.rate, request.pitch)
            full_audio.write(seg_audio)
            
            # --- LÓGICA DE SINCRONIA ---
            if seg_events:
                # Caso ideal: Microsoft mandou tempos
                last = seg_events[-1]
                # Adiciona offset global
                for event in seg_events:
                    event['offset'] += global_offset
                    all_events.append(event)
                global_offset = last['offset'] + last['duration'] + 500_000 
            else:
                # Caso Fallback: Microsoft falhou
                logger.warning(f"Segmento {i} usando estimativa calibrada ({MS_PER_CHAR_ESTIMATE}ms/char).")
                
                # CALIBRAÇÃO AQUI:
                # Transforma ms em ticks (1ms = 10,000 ticks)
                ticks_per_char = MS_PER_CHAR_ESTIMATE * 10_000
                estimated_duration = len(seg) * ticks_per_char
                
                # Gera eventos simulados proporcionais
                simulated = estimate_word_timings(seg, global_offset, estimated_duration)
                all_events.extend(simulated)
                
                global_offset += estimated_duration + 500_000 # +50ms pausa

            await asyncio.sleep(0.1)

        grouped_captions = group_words_to_captions(all_events, max_words=5)
        srt_content = generate_srt_string(grouped_captions)
        
        if not srt_content:
             srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{request.text[:50]}..."

        return {
            "status": "success",
            "data": {
                "audio_base64": base64.b64encode(full_audio.getvalue()).decode('utf-8'),
                "srt_base64": base64.b64encode(srt_content.encode('utf-8')).decode('utf-8'),
                "metadata": {"mode": "calibrated", "ms_per_char": MS_PER_CHAR_ESTIMATE}
            }
        }

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
