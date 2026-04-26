import asyncio
import io
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

try:
    from .settings import (
        APP_DIR,
        TEXT_AUDIO_APP_HOST,
        TEXT_AUDIO_APP_PORT,
        TEXT_AUDIO_MAX_CHARS,
        TEXT_AUDIO_TTS_API_KEY,
        TEXT_AUDIO_TTS_BASE_URL,
        TEXT_AUDIO_TTS_CHUNK_CHARS,
        TEXT_AUDIO_TTS_DEFAULT_VOICE,
        TEXT_AUDIO_TTS_POLL_ATTEMPTS,
        TEXT_AUDIO_TTS_POLL_SECONDS,
        validate_runtime_settings,
    )
except ImportError:
    from settings import (
        APP_DIR,
        TEXT_AUDIO_APP_HOST,
        TEXT_AUDIO_APP_PORT,
        TEXT_AUDIO_MAX_CHARS,
        TEXT_AUDIO_TTS_API_KEY,
        TEXT_AUDIO_TTS_BASE_URL,
        TEXT_AUDIO_TTS_CHUNK_CHARS,
        TEXT_AUDIO_TTS_DEFAULT_VOICE,
        TEXT_AUDIO_TTS_POLL_ATTEMPTS,
        TEXT_AUDIO_TTS_POLL_SECONDS,
        validate_runtime_settings,
    )


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [text-audio] %(message)s",
)
logger = logging.getLogger("text-audio")


app = FastAPI(
    title="Text to Audio Converter",
    description="Convert typed text into downloadable MP3 audio using the live CodeVoice API.",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


class VoiceOption(BaseModel):
    id: str
    label: str
    language: str
    language_name: str
    gender: str
    description: str
    quality: str
    rating: int
    downloads: int


class VoicesResponse(BaseModel):
    voices: List[VoiceOption]
    default_voice: str


class ConvertTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: Optional[str] = None
    language: Optional[str] = None


class ConvertTextResponse(BaseModel):
    status: str
    audio_url: str
    extracted_characters: int
    chunks_processed: int
    filename: str
    voice: str


QUALITY_PRIORITY = {
    "ultra": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}
TOOL_ROUTE_PREFIX = "/text-to-audio"


def read_html_file(name: str) -> str:
    path = APP_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return path.read_text(encoding="utf-8")


def normalize_voice_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_text_input(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def make_log_preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def build_voice_option(voice: dict) -> VoiceOption:
    voice_name = normalize_voice_value(str(voice.get("name") or voice.get("id") or ""))
    return VoiceOption(
        id=voice_name,
        label=voice_name,
        language=str(voice.get("language") or "").strip(),
        language_name=str(voice.get("language_name") or "Unknown").strip(),
        gender=str(voice.get("gender") or "unknown").strip(),
        description=str(voice.get("description") or f"Voice: {voice_name}").strip(),
        quality=str(voice.get("quality") or "unknown").strip(),
        rating=int(voice.get("rating") or 0),
        downloads=int(voice.get("downloads") or 0),
    )


def sort_voice_options(voices: List[VoiceOption]) -> List[VoiceOption]:
    return sorted(
        voices,
        key=lambda voice: (
            -voice.rating,
            -QUALITY_PRIORITY.get(voice.quality.lower(), 0),
            -voice.downloads,
            voice.label.lower(),
        ),
    )


async def fetch_available_tts_voices(client: httpx.AsyncClient) -> List[VoiceOption]:
    validate_runtime_settings()
    headers = {}
    if TEXT_AUDIO_TTS_API_KEY:
        headers["X-API-Key"] = TEXT_AUDIO_TTS_API_KEY

    response = await client.get(f"{TEXT_AUDIO_TTS_BASE_URL}/v1/voices", headers=headers)
    response.raise_for_status()
    payload = response.json()
    voices_payload = payload.get("voices")
    if not isinstance(voices_payload, list):
        raise RuntimeError("The TTS voices endpoint returned an invalid payload.")

    voices = [
        build_voice_option(voice)
        for voice in voices_payload
        if normalize_voice_value(str(voice.get("name") or voice.get("id") or ""))
    ]
    if not voices:
        raise RuntimeError("No TTS voices are currently available.")
    return sort_voice_options(voices)


def get_default_voice(voices: List[VoiceOption]) -> str:
    configured = normalize_voice_value(TEXT_AUDIO_TTS_DEFAULT_VOICE)
    if configured:
        for voice in voices:
            if voice.label.lower() == configured.lower() or voice.id.lower() == configured.lower():
                return voice.id
    return voices[0].id if voices else configured


def get_fallback_voice_option() -> VoiceOption:
    fallback_voice = normalize_voice_value(TEXT_AUDIO_TTS_DEFAULT_VOICE) or "Ryan"
    return VoiceOption(
        id=fallback_voice,
        label=fallback_voice,
        language="",
        language_name="Unknown",
        gender="unknown",
        description="Fallback voice configured in the text-to-audio tool.",
        quality="unknown",
        rating=0,
        downloads=0,
    )


def chunk_text_for_tts(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > TEXT_AUDIO_TTS_CHUNK_CHARS:
            parts = re.split(r"(?<=[,;:])\s+", sentence)
        else:
            parts = [sentence]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > TEXT_AUDIO_TTS_CHUNK_CHARS:
                for index in range(0, len(part), TEXT_AUDIO_TTS_CHUNK_CHARS):
                    segment = part[index:index + TEXT_AUDIO_TTS_CHUNK_CHARS].strip()
                    if not segment:
                        continue
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(segment)
                continue

            tentative = f"{current} {part}".strip() if current else part
            if len(tentative) <= TEXT_AUDIO_TTS_CHUNK_CHARS:
                current = tentative
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


async def create_tts_job(client: httpx.AsyncClient, text: str, voice: str) -> str:
    validate_runtime_settings()
    headers = {"Content-Type": "application/json"}
    if TEXT_AUDIO_TTS_API_KEY:
        headers["X-API-Key"] = TEXT_AUDIO_TTS_API_KEY

    response = await client.post(
        f"{TEXT_AUDIO_TTS_BASE_URL}/v1/tts",
        headers=headers,
        json={"text": text, "voice": voice},
    )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError("No job_id returned from the TTS API.")
    return job_id


async def wait_for_tts_audio(client: httpx.AsyncClient, job_id: str, request_id: str, chunk_index: int) -> Tuple[bytes, str]:
    headers = {}
    if TEXT_AUDIO_TTS_API_KEY:
        headers["X-API-Key"] = TEXT_AUDIO_TTS_API_KEY

    for attempt in range(1, TEXT_AUDIO_TTS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(TEXT_AUDIO_TTS_POLL_SECONDS)
        status_response = await client.get(f"{TEXT_AUDIO_TTS_BASE_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        logger.info(
            "[%s] Chunk %s poll %s/%s for job %s returned status=%s",
            request_id,
            chunk_index,
            attempt,
            TEXT_AUDIO_TTS_POLL_ATTEMPTS,
            job_id,
            status,
        )

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{TEXT_AUDIO_TTS_BASE_URL}/v1/audio/{job_id}", headers=headers)
            audio_response.raise_for_status()
            logger.info(
                "[%s] Chunk %s audio downloaded for job %s (%s bytes, format=%s)",
                request_id,
                chunk_index,
                job_id,
                len(audio_response.content),
                audio_format,
            )
            return audio_response.content, audio_format

        if status == "failed":
            raise RuntimeError(status_payload.get("error") or "The TTS job failed.")

    raise RuntimeError("The TTS request timed out before audio was ready.")


async def convert_text_to_audio(clean_text: str, request_id: str, voice: str) -> str:
    chunks = chunk_text_for_tts(clean_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The text is empty after cleanup.")

    merged_audio: Optional[AudioSegment] = None
    logger.info("[%s] Split text into %s chunk(s) for TTS", request_id, len(chunks))

    async with httpx.AsyncClient(timeout=180) as client:
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "[%s] Sending chunk %s/%s to TTS API %s/v1/tts (%s chars). Preview: \"%s\"",
                request_id,
                chunk_index,
                len(chunks),
                TEXT_AUDIO_TTS_BASE_URL,
                len(chunk),
                make_log_preview(chunk),
            )
            job_id = await create_tts_job(client, chunk, voice)
            logger.info("[%s] TTS job created for chunk %s: job_id=%s", request_id, chunk_index, job_id)
            audio_bytes, audio_format = await wait_for_tts_audio(client, job_id, request_id, chunk_index)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav" if audio_format == "wav" else "mp3")
            merged_audio = segment if merged_audio is None else merged_audio + segment
            logger.info(
                "[%s] Appended chunk %s/%s audio segment (%s ms)",
                request_id,
                chunk_index,
                len(chunks),
                len(segment),
            )

    stem_source = re.sub(r"[^a-zA-Z0-9_-]+", "-", make_log_preview(clean_text, limit=40)).strip("-").lower() or "text-audio"
    output_name = f"{stem_source}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    logger.info("[%s] Exporting merged audio to %s", request_id, output_path)
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    logger.info("[%s] Finished MP3 export: %s", request_id, output_name)
    return output_name


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def text_to_audio_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("text-to-audio.html"))


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/voices", response_model=VoicesResponse)
async def list_tts_voices() -> VoicesResponse:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            voices = await fetch_available_tts_voices(client)
        return VoicesResponse(voices=voices, default_voice=get_default_voice(voices))
    except Exception as exc:
        logger.warning("Unable to load live CodeVoice voices (%s). Falling back to configured default.", exc)
        fallback = get_fallback_voice_option()
        return VoicesResponse(voices=[fallback], default_voice=fallback.id)


@app.post("/convert-text", response_model=ConvertTextResponse)
async def convert_text(request: Request, payload: ConvertTextRequest) -> ConvertTextResponse:
    request_id = uuid.uuid4().hex[:8]
    cleaned_text = normalize_text_input(payload.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Please enter some text before converting.")
    if len(cleaned_text) > TEXT_AUDIO_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long. The current limit is {TEXT_AUDIO_MAX_CHARS:,} characters.",
        )

    requested_voice = normalize_voice_value(payload.voice or TEXT_AUDIO_TTS_DEFAULT_VOICE)
    if not requested_voice:
        raise HTTPException(status_code=400, detail="Please select a voice before converting.")

    logger.info("[%s] Received text-to-audio request (%s chars)", request_id, len(cleaned_text))

    try:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                available_voices = await fetch_available_tts_voices(client)
            matching_voice = next(
                (item for item in available_voices if item.id.lower() == requested_voice.lower() or item.label.lower() == requested_voice.lower()),
                None,
            )
            if not matching_voice:
                raise HTTPException(status_code=400, detail="The selected voice is not currently available.")
            selected_voice = matching_voice.id
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("[%s] Voice validation lookup failed (%s). Using requested voice directly.", request_id, exc)
            selected_voice = requested_voice

        logger.info("[%s] Using TTS voice: %s", request_id, selected_voice)
        output_name = await convert_text_to_audio(cleaned_text, request_id, selected_voice)
        output_url = str(request.base_url).rstrip("/") + f"{TOOL_ROUTE_PREFIX}/outputs/{output_name}"
        logger.info("[%s] Conversion complete. Audio available at %s", request_id, output_url)
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("[%s] Conversion failed during TTS processing", request_id)
        detail = str(exc)
        status_code = 504 if "timed out" in detail.lower() else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Conversion failed while calling the TTS service", request_id)
        raise HTTPException(status_code=502, detail="Failed to reach the configured TTS service.") from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected conversion error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during text conversion.") from exc

    return ConvertTextResponse(
        status="success",
        audio_url=output_url,
        extracted_characters=len(cleaned_text),
        chunks_processed=len(chunk_text_for_tts(cleaned_text)),
        filename=output_name,
        voice=selected_voice,
    )


@app.get("/download/{filename}", include_in_schema=False)
async def download_output(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    logger.info("Serving audio download: %s", safe_name)
    return FileResponse(file_path, media_type="audio/mpeg", filename=safe_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("texttoaudio:app", host=TEXT_AUDIO_APP_HOST, port=TEXT_AUDIO_APP_PORT, reload=True)
