import asyncio
import io
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import httpx
import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pydub import AudioSegment

try:
    from .settings import (
        APP_DIR,
        APP_HOST,
        APP_PORT,
        CLEANER_API_KEY,
        CLEANER_API_URL,
        CLEANER_MODEL,
        MAX_FILE_SIZE,
        MAX_TEXT_CHARS,
        TTS_API_KEY,
        TTS_BASE_URL,
        TTS_CHUNK_CHARS,
        TTS_DEFAULT_VOICE,
        TTS_POLL_ATTEMPTS,
        TTS_POLL_SECONDS,
        validate_runtime_settings,
    )
except ImportError:
    from settings import (
        APP_DIR,
        APP_HOST,
        APP_PORT,
        CLEANER_API_KEY,
        CLEANER_API_URL,
        CLEANER_MODEL,
        MAX_FILE_SIZE,
        MAX_TEXT_CHARS,
        TTS_API_KEY,
        TTS_BASE_URL,
        TTS_CHUNK_CHARS,
        TTS_DEFAULT_VOICE,
        TTS_POLL_ATTEMPTS,
        TTS_POLL_SECONDS,
        validate_runtime_settings,
    )


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pdf-audio] %(message)s",
)
logger = logging.getLogger("pdf-audio")


app = FastAPI(
    title="PDF to Audio Converter",
    description="Upload a PDF, extract text, clean it, and turn it into downloadable audio.",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


class ConvertResponse(BaseModel):
    status: str
    audio_url: str
    extracted_characters: int
    chunks_processed: int
    filename: str
    voice: str


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


QUALITY_PRIORITY = {
    "ultra": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def read_html_file(name: str) -> str:
    path = APP_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return path.read_text(encoding="utf-8")


def strip_code_fences(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def local_cleanup_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    filtered: List[str] = []
    for line in lines:
        if not line:
            filtered.append("")
            continue
        if re.fullmatch(r"(page\s*)?\d+", line, flags=re.IGNORECASE):
            continue
        if len(line) <= 2 and not any(ch.isalpha() for ch in line):
            continue
        filtered.append(line)

    cleaned = "\n".join(filtered)
    cleaned = re.sub(r"-\s*\n(?=\w)", "", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def make_log_preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def normalize_voice_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


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
    if TTS_API_KEY:
        headers["X-API-Key"] = TTS_API_KEY

    response = await client.get(f"{TTS_BASE_URL}/v1/voices", headers=headers)
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
    configured = normalize_voice_value(TTS_DEFAULT_VOICE)
    if configured:
        for voice in voices:
            if voice.label.lower() == configured.lower() or voice.id.lower() == configured.lower():
                return voice.id
    return voices[0].id if voices else configured


def get_fallback_voice_option() -> VoiceOption:
    fallback_voice = normalize_voice_value(TTS_DEFAULT_VOICE) or "ryan"
    return VoiceOption(
        id=fallback_voice,
        label=fallback_voice,
        language="",
        language_name="Unknown",
        gender="unknown",
        description="Fallback voice configured in the adsense tool.",
        quality="unknown",
        rating=0,
        downloads=0,
    )


def extract_text_from_pdf_bytes(file_bytes: bytes, request_id: str) -> str:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read the uploaded PDF: {exc}") from exc

    extracted_pages: List[str] = []
    ocr_pages: List[str] = []

    try:
        logger.info("[%s] Opened PDF with %s page(s)", request_id, document.page_count)
        for page in document:
            page_number = page.number + 1
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_pages.append(page_text)
                logger.info(
                    "[%s] Extracted text from page %s (%s chars)",
                    request_id,
                    page_number,
                    len(page_text),
                )
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                ocr_pages.append(ocr_text)
                logger.info(
                    "[%s] OCR extracted text from page %s (%s chars)",
                    request_id,
                    page_number,
                    len(ocr_text),
                )
            else:
                logger.info("[%s] No readable text found on page %s", request_id, page_number)
    finally:
        document.close()

    text = "\n\n".join(part for part in extracted_pages + ocr_pages if part.strip())
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="We couldn't extract readable text from this PDF. Try a clearer PDF or enable OCR on the source file.",
        )
    logger.info(
        "[%s] Combined extracted text: %s chars from %s direct page(s) and %s OCR page(s)",
        request_id,
        len(text),
        len(extracted_pages),
        len(ocr_pages),
    )
    return text


async def clean_text_with_internal_ai(text: str, request_id: str) -> str:
    trimmed = text[:MAX_TEXT_CHARS].strip()
    if not trimmed:
        raise HTTPException(status_code=422, detail="The PDF did not contain enough readable text to convert.")

    if not (CLEANER_API_URL and CLEANER_API_KEY):
        logger.info("[%s] Cleaner API not configured, using local cleanup", request_id)
        return local_cleanup_text(trimmed)

    prompt = (
        "Clean the following PDF text for text-to-speech. Remove repeated headers, footers, page numbers, "
        "broken line wraps, and OCR noise. Preserve meaning. Return plain text only.\n\n"
        f"{trimmed}"
    )

    headers = {
        "Authorization": f"Bearer {CLEANER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CLEANER_MODEL or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You clean extracted PDF text for natural speech synthesis."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            logger.info(
                "[%s] Sending cleaned-text request to %s with %s trimmed chars. Preview: \"%s\"",
                request_id,
                CLEANER_API_URL,
                len(trimmed),
                make_log_preview(trimmed),
            )
            response = await client.post(CLEANER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info("[%s] Cleaner API responded with status %s", request_id, response.status_code)
    except Exception as exc:
        logger.warning("[%s] Cleaner API failed (%s). Falling back to local cleanup", request_id, exc)
        return local_cleanup_text(trimmed)

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    cleaned = strip_code_fences(content)
    if cleaned:
        logger.info("[%s] Cleaner API returned %s cleaned chars", request_id, len(cleaned))
        return cleaned

    logger.info("[%s] Cleaner API returned empty text, using local cleanup fallback", request_id)
    return local_cleanup_text(trimmed)


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

        if len(sentence) > TTS_CHUNK_CHARS:
            parts = re.split(r"(?<=[,;:])\s+", sentence)
        else:
            parts = [sentence]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > TTS_CHUNK_CHARS:
                for index in range(0, len(part), TTS_CHUNK_CHARS):
                    segment = part[index:index + TTS_CHUNK_CHARS].strip()
                    if not segment:
                        continue
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(segment)
                continue

            tentative = f"{current} {part}".strip() if current else part
            if len(tentative) <= TTS_CHUNK_CHARS:
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
    if TTS_API_KEY:
        headers["X-API-Key"] = TTS_API_KEY

    response = await client.post(
        f"{TTS_BASE_URL}/v1/tts",
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
    if TTS_API_KEY:
        headers["X-API-Key"] = TTS_API_KEY

    for attempt in range(1, TTS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(TTS_POLL_SECONDS)
        status_response = await client.get(f"{TTS_BASE_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        logger.info(
            "[%s] Chunk %s poll %s/%s for job %s returned status=%s",
            request_id,
            chunk_index,
            attempt,
            TTS_POLL_ATTEMPTS,
            job_id,
            status,
        )

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{TTS_BASE_URL}/v1/audio/{job_id}", headers=headers)
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


async def convert_text_to_audio(clean_text: str, source_name: str, request_id: str, voice: str) -> str:
    chunks = chunk_text_for_tts(clean_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The cleaned PDF text was empty after processing.")

    merged_audio: Optional[AudioSegment] = None
    logger.info("[%s] Split cleaned text into %s chunk(s) for TTS", request_id, len(chunks))

    async with httpx.AsyncClient(timeout=180) as client:
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "[%s] Sending chunk %s/%s to TTS API %s/v1/tts (%s chars). Preview: \"%s\"",
                request_id,
                chunk_index,
                len(chunks),
                TTS_BASE_URL,
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

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(source_name).stem).strip("-").lower() or "pdf-audio"
    output_name = f"{stem}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    logger.info("[%s] Exporting merged audio to %s", request_id, output_path)
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    logger.info("[%s] Finished MP3 export: %s", request_id, output_name)
    return output_name


@app.get("/pdf-to-audio", response_class=HTMLResponse, include_in_schema=False)
async def pdf_to_audio_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("pdf-to-audio.html"))


@app.get("/pdf-to-audio.html", response_class=HTMLResponse, include_in_schema=False)
async def pdf_to_audio_page_html() -> HTMLResponse:
    return HTMLResponse(read_html_file("pdf-to-audio.html"))


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


@app.post("/convert-pdf", response_model=ConvertResponse)
async def convert_pdf(request: Request, voice: Optional[str] = Form(None), file: UploadFile = File(...)) -> ConvertResponse:
    request_id = uuid.uuid4().hex[:8]
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a PDF file before converting.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    logger.info("[%s] Received PDF upload: %s", request_id, file.filename)
    requested_voice = normalize_voice_value(voice or TTS_DEFAULT_VOICE)
    if not requested_voice:
        raise HTTPException(status_code=400, detail="Please select a voice before converting.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File is too large. The maximum supported size is 10MB.")
    logger.info("[%s] Uploaded PDF size: %s bytes", request_id, len(file_bytes))

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

        extracted_text = await asyncio.to_thread(extract_text_from_pdf_bytes, file_bytes, request_id)
        logger.info("[%s] Extracted total text length: %s chars", request_id, len(extracted_text))
        cleaned_text = await clean_text_with_internal_ai(extracted_text, request_id)
        logger.info("[%s] Cleaned text length: %s chars", request_id, len(cleaned_text))
        output_name = await convert_text_to_audio(cleaned_text, file.filename, request_id, selected_voice)
        output_url = str(request.base_url).rstrip("/") + f"/outputs/{output_name}"
        logger.info("[%s] Conversion complete. Audio available at %s", request_id, output_url)
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("[%s] Conversion failed during TTS processing", request_id)
        detail = str(exc)
        status_code = 504 if "timed out" in detail.lower() else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Conversion failed while calling an upstream HTTP service", request_id)
        raise HTTPException(
            status_code=502,
            detail="Failed to reach the configured text cleanup or TTS service.",
        ) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected conversion error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during PDF conversion.") from exc

    return ConvertResponse(
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

    uvicorn.run("pdftoaudio:app", host=APP_HOST, port=APP_PORT, reload=True)
