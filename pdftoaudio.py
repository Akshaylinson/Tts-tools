import asyncio
import io
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import httpx
import pytesseract
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pydub import AudioSegment
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
    TTS_POLL_ATTEMPTS,
    TTS_POLL_SECONDS,
    TTS_VOICE,
    validate_runtime_settings,
)


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read the uploaded PDF: {exc}") from exc

    extracted_pages: List[str] = []
    ocr_pages: List[str] = []

    try:
        for page in document:
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_pages.append(page_text)
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                ocr_pages.append(ocr_text)
    finally:
        document.close()

    text = "\n\n".join(part for part in extracted_pages + ocr_pages if part.strip())
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="We couldn't extract readable text from this PDF. Try a clearer PDF or enable OCR on the source file.",
        )
    return text


async def clean_text_with_internal_ai(text: str) -> str:
    trimmed = text[:MAX_TEXT_CHARS].strip()
    if not trimmed:
        raise HTTPException(status_code=422, detail="The PDF did not contain enough readable text to convert.")

    if not (CLEANER_API_URL and CLEANER_API_KEY):
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
            response = await client.post(CLEANER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return local_cleanup_text(trimmed)

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    cleaned = strip_code_fences(content)
    return cleaned or local_cleanup_text(trimmed)


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


async def create_tts_job(client: httpx.AsyncClient, text: str) -> str:
    validate_runtime_settings()
    headers = {"Content-Type": "application/json"}
    if TTS_API_KEY:
        headers["X-API-Key"] = TTS_API_KEY

    response = await client.post(
        f"{TTS_BASE_URL}/v1/tts",
        headers=headers,
        json={"text": text, "voice": TTS_VOICE},
    )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError("No job_id returned from the TTS API.")
    return job_id


async def wait_for_tts_audio(client: httpx.AsyncClient, job_id: str) -> Tuple[bytes, str]:
    headers = {}
    if TTS_API_KEY:
        headers["X-API-Key"] = TTS_API_KEY

    for _ in range(TTS_POLL_ATTEMPTS):
        await asyncio.sleep(TTS_POLL_SECONDS)
        status_response = await client.get(f"{TTS_BASE_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{TTS_BASE_URL}/v1/audio/{job_id}", headers=headers)
            audio_response.raise_for_status()
            return audio_response.content, audio_format

        if status == "failed":
            raise RuntimeError(status_payload.get("error") or "The TTS job failed.")

    raise RuntimeError("The TTS request timed out before audio was ready.")


async def convert_text_to_audio(clean_text: str, source_name: str) -> str:
    chunks = chunk_text_for_tts(clean_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The cleaned PDF text was empty after processing.")

    merged_audio: Optional[AudioSegment] = None

    async with httpx.AsyncClient(timeout=180) as client:
        for chunk in chunks:
            job_id = await create_tts_job(client, chunk)
            audio_bytes, audio_format = await wait_for_tts_audio(client, job_id)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav" if audio_format == "wav" else "mp3")
            merged_audio = segment if merged_audio is None else merged_audio + segment

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(source_name).stem).strip("-").lower() or "pdf-audio"
    output_name = f"{stem}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    return output_name


@app.get("/", include_in_schema=False)
async def home() -> HTMLResponse:
    return HTMLResponse(read_html_file("index.html"))


@app.get("/pdf-to-audio", response_class=HTMLResponse, include_in_schema=False)
async def pdf_to_audio_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("pdf-to-audio.html"))


@app.get("/pdf-to-audio.html", response_class=HTMLResponse, include_in_schema=False)
async def pdf_to_audio_page_html() -> HTMLResponse:
    return HTMLResponse(read_html_file("pdf-to-audio.html"))


@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
async def home_html() -> HTMLResponse:
    return HTMLResponse(read_html_file("index.html"))


@app.get("/privacy-policy", response_class=HTMLResponse, include_in_schema=False)
async def privacy_policy_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("privacy-policy.html"))


@app.get("/terms", response_class=HTMLResponse, include_in_schema=False)
async def terms_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("terms.html"))


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
async def about_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("about.html"))


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


@app.post("/convert-pdf", response_model=ConvertResponse)
async def convert_pdf(request: Request, file: UploadFile = File(...)) -> ConvertResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a PDF file before converting.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File is too large. The maximum supported size is 10MB.")

    extracted_text = await asyncio.to_thread(extract_text_from_pdf_bytes, file_bytes)
    cleaned_text = await clean_text_with_internal_ai(extracted_text)
    output_name = await convert_text_to_audio(cleaned_text, file.filename)

    return ConvertResponse(
        status="success",
        audio_url=str(request.base_url).rstrip("/") + f"/outputs/{output_name}",
        extracted_characters=len(cleaned_text),
        chunks_processed=len(chunk_text_for_tts(cleaned_text)),
        filename=output_name,
    )


@app.get("/download/{filename}", include_in_schema=False)
async def download_output(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(file_path, media_type="audio/mpeg", filename=safe_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pdftoaudio:app", host=APP_HOST, port=APP_PORT, reload=True)
