import asyncio
import io
import ipaddress
import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment
from readability import Document

try:
    from .settings import (
        APP_DIR,
        BLOG_FETCH_MAX_CHARS,
        BLOG_FETCH_TIMEOUT_SECONDS,
        BLOG_GROK_API_KEY,
        BLOG_GROK_BASE_URL,
        BLOG_GROK_MODEL,
        BLOG_GROK_TIMEOUT_SECONDS,
        BLOG_MAX_INPUT_CHARS,
        BLOG_PODCAST_APP_HOST,
        BLOG_PODCAST_APP_PORT,
        BLOG_TTS_API_KEY,
        BLOG_TTS_BASE_URL,
        BLOG_TTS_CHUNK_CHARS,
        BLOG_TTS_DEFAULT_VOICE,
        BLOG_TTS_POLL_ATTEMPTS,
        BLOG_TTS_POLL_SECONDS,
        validate_grok_runtime_settings,
        validate_tts_runtime_settings,
    )
except ImportError:
    from settings import (
        APP_DIR,
        BLOG_FETCH_MAX_CHARS,
        BLOG_FETCH_TIMEOUT_SECONDS,
        BLOG_GROK_API_KEY,
        BLOG_GROK_BASE_URL,
        BLOG_GROK_MODEL,
        BLOG_GROK_TIMEOUT_SECONDS,
        BLOG_MAX_INPUT_CHARS,
        BLOG_PODCAST_APP_HOST,
        BLOG_PODCAST_APP_PORT,
        BLOG_TTS_API_KEY,
        BLOG_TTS_BASE_URL,
        BLOG_TTS_CHUNK_CHARS,
        BLOG_TTS_DEFAULT_VOICE,
        BLOG_TTS_POLL_ATTEMPTS,
        BLOG_TTS_POLL_SECONDS,
        validate_grok_runtime_settings,
        validate_tts_runtime_settings,
    )


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("blog-to-podcast")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [blog-to-podcast] %(message)s"))
    logger.addHandler(handler)

app = FastAPI(
    title="Blog to Podcast Converter",
    description="Convert article URLs or pasted text into podcast-style scripts and downloadable audio.",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

TOOL_ROUTE_PREFIX = "/blog-to-podcast"
QUALITY_PRIORITY = {
    "ultra": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}
LENGTH_WORD_TARGETS = {
    "short": (120, 200),
    "medium": (300, 600),
    "full": (700, 1200),
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 BlogToPodcastBot/1.0"
)


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


class ExtractContentRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class ExtractContentResponse(BaseModel):
    text: str
    source_url: str
    extracted_characters: int


class GeneratePodcastRequest(BaseModel):
    text: str = Field(..., min_length=30, max_length=24000)
    length: str = Field(..., min_length=5, max_length=16)
    language: str = Field(..., min_length=2, max_length=64)


class PodcastScriptPayload(BaseModel):
    title: str
    intro: str
    main_content: List[str]
    outro: str
    narration_text: str
    estimated_word_count: int


class GeneratePodcastResponse(BaseModel):
    status: str
    podcast: PodcastScriptPayload
    length: str
    language: str


class GenerateAudioRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=16000)
    voice_model: str = Field(..., min_length=1, max_length=100)


class GenerateAudioResponse(BaseModel):
    status: str
    audio_url: str
    filename: str
    chunks_processed: int
    extracted_characters: int
    voice: str


def read_html_file(name: str) -> str:
    path = APP_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return path.read_text(encoding="utf-8")


def normalize_text_input(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_voice_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


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
    validate_tts_runtime_settings()
    headers = {}
    if BLOG_TTS_API_KEY:
        headers["X-API-Key"] = BLOG_TTS_API_KEY

    response = await client.get(f"{BLOG_TTS_BASE_URL}/v1/voices", headers=headers)
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
    configured = normalize_voice_value(BLOG_TTS_DEFAULT_VOICE)
    if configured:
        for voice in voices:
            if voice.id.lower() == configured.lower() or voice.label.lower() == configured.lower():
                return voice.id
    return voices[0].id if voices else configured


def get_fallback_voice_option() -> VoiceOption:
    fallback_voice = normalize_voice_value(BLOG_TTS_DEFAULT_VOICE) or "Ryan"
    return VoiceOption(
        id=fallback_voice,
        label=fallback_voice,
        language="",
        language_name="Unknown",
        gender="unknown",
        description="Fallback voice configured for the blog to podcast converter.",
        quality="unknown",
        rating=0,
        downloads=0,
    )


def validate_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Please enter a valid public article URL.")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="Please enter a valid public article URL.")
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        raise HTTPException(status_code=400, detail="Local or private URLs are not allowed.")
    try:
        ip_value = ipaddress.ip_address(hostname)
    except ValueError:
        ip_value = None
    if ip_value and (ip_value.is_private or ip_value.is_loopback or ip_value.is_link_local or ip_value.is_reserved):
        raise HTTPException(status_code=400, detail="Local or private URLs are not allowed.")
    return parsed.geturl()


def clean_extracted_text(text: str) -> str:
    lines = [line.strip() for line in normalize_text_input(text).splitlines()]
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        if len(line) < 25 and lower_line in {
            "advertisement",
            "sign up",
            "subscribe",
            "cookie policy",
            "related articles",
            "share this article",
        }:
            continue
        cleaned_lines.append(line)
    cleaned = "\n\n".join(line for line in cleaned_lines if line)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) > BLOG_FETCH_MAX_CHARS:
        cleaned = cleaned[:BLOG_FETCH_MAX_CHARS].rsplit(" ", 1)[0].strip()
    return cleaned.strip()


def extract_text_from_html(html: str) -> str:
    doc = Document(html)
    readable_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(readable_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "aside", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    cleaned = clean_extracted_text(text)
    if cleaned:
        return cleaned

    fallback_soup = BeautifulSoup(html, "html.parser")
    for tag in fallback_soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return clean_extracted_text(fallback_soup.get_text("\n", strip=True))


async def fetch_article_text(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(
        timeout=BLOG_FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise RuntimeError("The provided URL did not return a standard article page.")
        return extract_text_from_html(response.text)


def extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError("The podcast model did not return a valid JSON object.")
        return json.loads(match.group(0))


def clean_required_section(value: str, label: str) -> str:
    cleaned = normalize_text_input(value)
    if not cleaned:
        raise RuntimeError(f"The podcast response is missing the {label} section.")
    return cleaned


def clean_main_content(value: object) -> List[str]:
    segments: List[str] = []
    if isinstance(value, list):
        segments = [normalize_text_input(str(item)) for item in value]
    elif isinstance(value, str):
        parts = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-Z])", value)
        segments = [normalize_text_input(part) for part in parts]
    cleaned = [segment for segment in segments if segment]
    if not cleaned:
        raise RuntimeError("The podcast response is missing the main content section.")
    return cleaned


def compose_narration_text(title: str, intro: str, main_content: List[str], outro: str) -> str:
    joined_main = " ... ".join(main_content)
    return normalize_text_input(f"{title}. ... {intro} ... {joined_main} ... {outro}")


def build_podcast_prompt(payload: GeneratePodcastRequest) -> Tuple[str, str]:
    min_words, max_words = LENGTH_WORD_TARGETS[payload.length]
    system_prompt = (
        "You are an expert podcast scriptwriter. "
        "Transform articles into polished single-speaker podcast narration that sounds natural when spoken aloud. "
        "Return valid JSON only with these keys: title, intro, main_content, outro. "
        "main_content must be an array of 2 to 5 spoken-friendly segments. "
        "Do not use markdown, bullet lists, or stage directions."
    )
    user_prompt = (
        f"Language: {payload.language}\n"
        f"Target length: {payload.length}\n"
        f"Word budget: {min_words} to {max_words} words\n\n"
        "Transformation rules:\n"
        "- Write for audio, not for reading.\n"
        "- Open with a compelling intro that quickly frames the article.\n"
        "- Keep one narrator voice throughout.\n"
        "- Simplify dense sentences and explain ideas smoothly.\n"
        "- Preserve the key insights, examples, and takeaways from the source.\n"
        "- Use clear transitions between segments.\n"
        "- End with a concise outro.\n"
        "- Avoid filler, clickbait, direct quoting, and unsupported claims.\n"
        "- Keep it safe for a general audience and AdSense-friendly.\n\n"
        f"Source article text:\n{payload.text}\n\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


async def generate_podcast_from_grok(payload: GeneratePodcastRequest, request_id: str) -> PodcastScriptPayload:
    validate_grok_runtime_settings()
    system_prompt, user_prompt = build_podcast_prompt(payload)
    headers = {
        "Authorization": f"Bearer {BLOG_GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": BLOG_GROK_MODEL,
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=BLOG_GROK_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{BLOG_GROK_BASE_URL}/chat/completions", headers=headers, json=body)
        if response.is_error:
            logger.error(
                "[%s] Grok upstream error %s from %s: %s",
                request_id,
                response.status_code,
                response.request.url,
                response.text,
            )
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("The podcast model returned no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The podcast model returned an empty response.")

    parsed = extract_json_object(content)
    title = clean_required_section(str(parsed.get("title") or ""), "title")
    intro = clean_required_section(str(parsed.get("intro") or ""), "intro")
    main_content = clean_main_content(parsed.get("main_content"))
    outro = clean_required_section(str(parsed.get("outro") or ""), "outro")
    narration_text = compose_narration_text(title, intro, main_content, outro)
    estimated_word_count = len(narration_text.split())

    return PodcastScriptPayload(
        title=title,
        intro=intro,
        main_content=main_content,
        outro=outro,
        narration_text=narration_text,
        estimated_word_count=estimated_word_count,
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

        parts = re.split(r"(?<=[,;:])\s+", sentence) if len(sentence) > BLOG_TTS_CHUNK_CHARS else [sentence]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > BLOG_TTS_CHUNK_CHARS:
                for index in range(0, len(part), BLOG_TTS_CHUNK_CHARS):
                    segment = part[index:index + BLOG_TTS_CHUNK_CHARS].strip()
                    if not segment:
                        continue
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(segment)
                continue

            tentative = f"{current} {part}".strip() if current else part
            if len(tentative) <= BLOG_TTS_CHUNK_CHARS:
                current = tentative
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


async def create_tts_job(client: httpx.AsyncClient, text: str, voice: str) -> str:
    validate_tts_runtime_settings()
    headers = {"Content-Type": "application/json"}
    if BLOG_TTS_API_KEY:
        headers["X-API-Key"] = BLOG_TTS_API_KEY

    response = await client.post(
        f"{BLOG_TTS_BASE_URL}/v1/tts",
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
    if BLOG_TTS_API_KEY:
        headers["X-API-Key"] = BLOG_TTS_API_KEY

    for attempt in range(1, BLOG_TTS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(BLOG_TTS_POLL_SECONDS)
        status_response = await client.get(f"{BLOG_TTS_BASE_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        logger.info(
            "[%s] Chunk %s poll %s/%s for job %s returned status=%s",
            request_id,
            chunk_index,
            attempt,
            BLOG_TTS_POLL_ATTEMPTS,
            job_id,
            status,
        )

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{BLOG_TTS_BASE_URL}/v1/audio/{job_id}", headers=headers)
            audio_response.raise_for_status()
            return audio_response.content, audio_format

        if status == "failed":
            raise RuntimeError(status_payload.get("error") or "The TTS job failed.")

    raise RuntimeError("The TTS request timed out before audio was ready.")


async def convert_text_to_audio(clean_text: str, request_id: str, voice: str) -> str:
    chunks = chunk_text_for_tts(clean_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The narration text is empty after cleanup.")

    merged_audio: Optional[AudioSegment] = None
    logger.info("[%s] Split narration into %s chunk(s)", request_id, len(chunks))

    async with httpx.AsyncClient(timeout=180) as client:
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "[%s] Sending chunk %s/%s to TTS API %s/v1/tts (%s chars). Preview: \"%s\"",
                request_id,
                chunk_index,
                len(chunks),
                BLOG_TTS_BASE_URL,
                len(chunk),
                make_log_preview(chunk),
            )
            job_id = await create_tts_job(client, chunk, voice)
            audio_bytes, audio_format = await wait_for_tts_audio(client, job_id, request_id, chunk_index)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav" if audio_format == "wav" else "mp3")
            merged_audio = segment if merged_audio is None else merged_audio + segment

    stem_source = re.sub(r"[^a-zA-Z0-9_-]+", "-", make_log_preview(clean_text, limit=40)).strip("-").lower() or "podcast"
    output_name = f"{stem_source}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    return output_name


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def blog_to_podcast_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("blog-to-podcast.html"))


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
        logger.warning("Unable to load live TTS voices (%s). Falling back to configured default.", exc)
        fallback = get_fallback_voice_option()
        return VoicesResponse(voices=[fallback], default_voice=fallback.id)


@app.post("/extract-content", response_model=ExtractContentResponse)
async def extract_content(payload: ExtractContentRequest) -> ExtractContentResponse:
    url = validate_public_url(payload.url)
    try:
        extracted_text = await fetch_article_text(url)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Article fetch failed with status {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to fetch the article URL right now.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not extracted_text or len(extracted_text) < 120:
        raise HTTPException(status_code=422, detail="We could not extract enough readable article text from that URL.")

    return ExtractContentResponse(
        text=extracted_text,
        source_url=url,
        extracted_characters=len(extracted_text),
    )


@app.post("/generate-podcast", response_model=GeneratePodcastResponse)
async def generate_podcast(payload: GeneratePodcastRequest) -> GeneratePodcastResponse:
    request_id = uuid.uuid4().hex[:8]
    cleaned_text = normalize_text_input(payload.text)
    length = payload.length.strip().lower()
    language = normalize_text_input(payload.language)

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Please enter article text before generating a podcast.")
    if len(cleaned_text) > BLOG_MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Input is too long. The current limit is {BLOG_MAX_INPUT_CHARS:,} characters.",
        )
    if length not in LENGTH_WORD_TARGETS:
        raise HTTPException(status_code=400, detail="Please choose a valid podcast length.")
    if not language:
        raise HTTPException(status_code=400, detail="Please choose a podcast language.")

    normalized_payload = GeneratePodcastRequest(text=cleaned_text, length=length, language=language)

    try:
        logger.info("[%s] Generating podcast length=%s language=%s via Grok", request_id, length, language)
        podcast = await generate_podcast_from_grok(normalized_payload, request_id)
    except httpx.HTTPStatusError as exc:
        logger.exception("[%s] Podcast generation failed with upstream status error", request_id)
        upstream_detail = exc.response.text.strip() if exc.response is not None and exc.response.text else ""
        detail = "The podcast generation service returned an error."
        if "Model not found" in upstream_detail:
            detail = (
                "The configured model was not found. "
                "Update BLOG_GROQ_MODEL in tools/blog_to_podcast/.env to a Groq model your API key can access."
            )
        if "Incorrect API key provided" in upstream_detail:
            detail = (
                "The configured provider key was rejected. "
                "Set BLOG_GROQ_API_KEY in tools/blog_to_podcast/.env, or rely on STORY_GROQ_API_KEY as a shared key."
            )
        if upstream_detail:
            detail = f"{detail} Upstream response: {upstream_detail}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Podcast generation failed while calling Grok", request_id)
        raise HTTPException(status_code=502, detail="Unable to reach the podcast generation service right now.") from exc
    except RuntimeError as exc:
        logger.exception("[%s] Podcast generation returned invalid data", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected podcast generation error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during podcast generation.") from exc

    return GeneratePodcastResponse(
        status="success",
        podcast=podcast,
        length=length,
        language=language,
    )


@app.post("/generate-audio", response_model=GenerateAudioResponse)
async def generate_audio(request: Request, payload: GenerateAudioRequest) -> GenerateAudioResponse:
    request_id = uuid.uuid4().hex[:8]
    narration_text = normalize_text_input(payload.text)
    requested_voice = normalize_voice_value(payload.voice_model or BLOG_TTS_DEFAULT_VOICE)

    if not narration_text:
        raise HTTPException(status_code=400, detail="Please generate a podcast script before converting it to audio.")
    if not requested_voice:
        raise HTTPException(status_code=400, detail="Please choose a voice model before converting.")

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

        output_name = await convert_text_to_audio(narration_text, request_id, selected_voice)
        output_url = str(request.base_url).rstrip("/") + f"{TOOL_ROUTE_PREFIX}/outputs/{output_name}"
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("[%s] Audio generation failed during TTS processing", request_id)
        detail = str(exc)
        status_code = 504 if "timed out" in detail.lower() else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Audio generation failed while calling the TTS service", request_id)
        raise HTTPException(status_code=502, detail="Failed to reach the configured TTS service.") from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected audio generation error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during audio generation.") from exc

    return GenerateAudioResponse(
        status="success",
        audio_url=output_url,
        filename=output_name,
        chunks_processed=len(chunk_text_for_tts(narration_text)),
        extracted_characters=len(narration_text),
        voice=selected_voice,
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

    uvicorn.run("blogtopodcast:app", host=BLOG_PODCAST_APP_HOST, port=BLOG_PODCAST_APP_PORT, reload=True)
