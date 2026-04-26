import asyncio
import io
import json
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
        STORY_APP_HOST,
        STORY_APP_PORT,
        STORY_GROK_API_KEY,
        STORY_GROK_BASE_URL,
        STORY_GROK_MODEL,
        STORY_GROK_TIMEOUT_SECONDS,
        STORY_MAX_IDEA_CHARS,
        STORY_TTS_API_KEY,
        STORY_TTS_BASE_URL,
        STORY_TTS_CHUNK_CHARS,
        STORY_TTS_DEFAULT_VOICE,
        STORY_TTS_POLL_ATTEMPTS,
        STORY_TTS_POLL_SECONDS,
        validate_story_runtime_settings,
        validate_tts_runtime_settings,
    )
except ImportError:
    from settings import (
        APP_DIR,
        STORY_APP_HOST,
        STORY_APP_PORT,
        STORY_GROK_API_KEY,
        STORY_GROK_BASE_URL,
        STORY_GROK_MODEL,
        STORY_GROK_TIMEOUT_SECONDS,
        STORY_MAX_IDEA_CHARS,
        STORY_TTS_API_KEY,
        STORY_TTS_BASE_URL,
        STORY_TTS_CHUNK_CHARS,
        STORY_TTS_DEFAULT_VOICE,
        STORY_TTS_POLL_ATTEMPTS,
        STORY_TTS_POLL_SECONDS,
        validate_story_runtime_settings,
        validate_tts_runtime_settings,
    )


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [story-narration] %(message)s",
)
logger = logging.getLogger("story-narration")

app = FastAPI(
    title="AI Story Narration Generator",
    description="Generate structured stories from rough ideas and convert them into narrated audio.",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

TOOL_ROUTE_PREFIX = "/story-narration-generator"
QUALITY_PRIORITY = {
    "ultra": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}
DURATION_WORD_COUNTS = {
    "1 min": 140,
    "3 min": 420,
    "5 min": 700,
}
STORY_TYPE_OPTIONS = {"kids", "horror", "motivational", "general"}


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


class GenerateStoryRequest(BaseModel):
    idea: str = Field(..., min_length=3, max_length=2000)
    story_type: str = Field(..., min_length=3, max_length=32)
    duration: str = Field(..., min_length=3, max_length=16)
    language: str = Field(..., min_length=2, max_length=64)
    audience: str = Field(default="general", min_length=3, max_length=32)


class StoryPayload(BaseModel):
    title: str
    intro: str
    story: str
    ending: str
    moral: str
    narration_text: str
    estimated_word_count: int


class GenerateStoryResponse(BaseModel):
    status: str
    story: StoryPayload
    story_type: str
    duration: str
    language: str


class GenerateAudioRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=12000)
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
    validate_tts_runtime_settings()
    headers = {}
    if STORY_TTS_API_KEY:
        headers["X-API-Key"] = STORY_TTS_API_KEY

    response = await client.get(f"{STORY_TTS_BASE_URL}/v1/voices", headers=headers)
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
    configured = normalize_voice_value(STORY_TTS_DEFAULT_VOICE)
    if configured:
        for voice in voices:
            if voice.label.lower() == configured.lower() or voice.id.lower() == configured.lower():
                return voice.id
    return voices[0].id if voices else configured


def get_fallback_voice_option() -> VoiceOption:
    fallback_voice = normalize_voice_value(STORY_TTS_DEFAULT_VOICE) or "Ryan"
    return VoiceOption(
        id=fallback_voice,
        label=fallback_voice,
        language="",
        language_name="Unknown",
        gender="unknown",
        description="Fallback voice configured for the story narration generator.",
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

        if len(sentence) > STORY_TTS_CHUNK_CHARS:
            parts = re.split(r"(?<=[,;:])\s+", sentence)
        else:
            parts = [sentence]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > STORY_TTS_CHUNK_CHARS:
                for index in range(0, len(part), STORY_TTS_CHUNK_CHARS):
                    segment = part[index:index + STORY_TTS_CHUNK_CHARS].strip()
                    if not segment:
                        continue
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(segment)
                continue

            tentative = f"{current} {part}".strip() if current else part
            if len(tentative) <= STORY_TTS_CHUNK_CHARS:
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
    if STORY_TTS_API_KEY:
        headers["X-API-Key"] = STORY_TTS_API_KEY

    response = await client.post(
        f"{STORY_TTS_BASE_URL}/v1/tts",
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
    if STORY_TTS_API_KEY:
        headers["X-API-Key"] = STORY_TTS_API_KEY

    for attempt in range(1, STORY_TTS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(STORY_TTS_POLL_SECONDS)
        status_response = await client.get(f"{STORY_TTS_BASE_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        logger.info(
            "[%s] Chunk %s poll %s/%s for job %s returned status=%s",
            request_id,
            chunk_index,
            attempt,
            STORY_TTS_POLL_ATTEMPTS,
            job_id,
            status,
        )

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{STORY_TTS_BASE_URL}/v1/audio/{job_id}", headers=headers)
            audio_response.raise_for_status()
            return audio_response.content, audio_format

        if status == "failed":
            raise RuntimeError(status_payload.get("error") or "The TTS job failed.")

    raise RuntimeError("The TTS request timed out before audio was ready.")


def build_story_prompt(payload: GenerateStoryRequest) -> Tuple[str, str]:
    word_target = DURATION_WORD_COUNTS[payload.duration]
    system_prompt = (
        "You are an expert story writer for spoken narration. "
        "Turn rough ideas into polished stories that sound natural when read aloud. "
        "Always return valid JSON only with these keys: title, intro, story, ending, moral. "
        "Each value must be plain text with no markdown. "
        "Use short spoken-friendly sentences, natural pauses, and clear transitions. "
        "Keep the tone appropriate for the requested story type and audience."
    )
    user_prompt = (
        f"Idea: {payload.idea}\n"
        f"Story type: {payload.story_type}\n"
        f"Duration target: {payload.duration}\n"
        f"Approximate word count: {word_target}\n"
        f"Language: {payload.language}\n"
        f"Audience: {payload.audience}\n\n"
        "Output rules:\n"
        "- Write everything in the requested language.\n"
        "- Create a compelling title.\n"
        "- Keep the intro to 2 to 4 sentences.\n"
        "- Make the main story the longest section.\n"
        "- End with a satisfying closing.\n"
        "- Provide a short moral or takeaway.\n"
        "- Use narration-friendly rhythm with occasional pause markers like ... where useful.\n"
        "- Avoid graphic violence, hate, explicit sexual content, or unsafe instructions.\n"
        "- Keep the total word count close to the target.\n"
        "- Return JSON only."
    )
    return system_prompt, user_prompt


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
            raise RuntimeError("The story model did not return a valid JSON object.")
        return json.loads(match.group(0))


def clean_story_section(value: str, fallback: str) -> str:
    cleaned = normalize_text_input(value)
    if not cleaned:
        raise RuntimeError(f"The story response is missing the {fallback} section.")
    return cleaned


def compose_narration_text(title: str, intro: str, story: str, ending: str, moral: str) -> str:
    return normalize_text_input(
        f"{title}. ... {intro} ... {story} ... {ending} ... Moral: {moral}"
    )


async def generate_story_from_grok(payload: GenerateStoryRequest, request_id: str) -> StoryPayload:
    validate_story_runtime_settings()
    system_prompt, user_prompt = build_story_prompt(payload)
    headers = {
        "Authorization": f"Bearer {STORY_GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": STORY_GROK_MODEL,
        "temperature": 0.8,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=STORY_GROK_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{STORY_GROK_BASE_URL}/v1/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("The story model returned no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The story model returned an empty response.")

    logger.info("[%s] Story model response received", request_id)
    parsed = extract_json_object(content)
    title = clean_story_section(str(parsed.get("title") or ""), "title")
    intro = clean_story_section(str(parsed.get("intro") or ""), "intro")
    story = clean_story_section(str(parsed.get("story") or ""), "story")
    ending = clean_story_section(str(parsed.get("ending") or ""), "ending")
    moral = clean_story_section(str(parsed.get("moral") or ""), "moral")
    narration_text = compose_narration_text(title, intro, story, ending, moral)
    estimated_word_count = len(narration_text.split())

    return StoryPayload(
        title=title,
        intro=intro,
        story=story,
        ending=ending,
        moral=moral,
        narration_text=narration_text,
        estimated_word_count=estimated_word_count,
    )


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
                STORY_TTS_BASE_URL,
                len(chunk),
                make_log_preview(chunk),
            )
            job_id = await create_tts_job(client, chunk, voice)
            audio_bytes, audio_format = await wait_for_tts_audio(client, job_id, request_id, chunk_index)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav" if audio_format == "wav" else "mp3")
            merged_audio = segment if merged_audio is None else merged_audio + segment

    stem_source = re.sub(r"[^a-zA-Z0-9_-]+", "-", make_log_preview(clean_text, limit=40)).strip("-").lower() or "story-audio"
    output_name = f"{stem_source}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    return output_name


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def story_narration_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("story-narration-generator.html"))


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


@app.post("/generate-story", response_model=GenerateStoryResponse)
async def generate_story(payload: GenerateStoryRequest) -> GenerateStoryResponse:
    request_id = uuid.uuid4().hex[:8]
    idea = normalize_text_input(payload.idea)
    story_type = payload.story_type.strip().lower()
    duration = payload.duration.strip()
    language = normalize_text_input(payload.language)
    audience = payload.audience.strip().lower() or "general"

    if not idea:
        raise HTTPException(status_code=400, detail="Please enter your story idea before generating.")
    if len(idea) > STORY_MAX_IDEA_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Idea is too long. The current limit is {STORY_MAX_IDEA_CHARS:,} characters.",
        )
    if story_type not in STORY_TYPE_OPTIONS:
        raise HTTPException(status_code=400, detail="Please choose a valid story type.")
    if duration not in DURATION_WORD_COUNTS:
        raise HTTPException(status_code=400, detail="Please choose a valid target duration.")
    if not language:
        raise HTTPException(status_code=400, detail="Please choose a story language.")

    normalized_payload = GenerateStoryRequest(
        idea=idea,
        story_type=story_type,
        duration=duration,
        language=language,
        audience=audience,
    )

    try:
        logger.info("[%s] Generating story for type=%s duration=%s language=%s", request_id, story_type, duration, language)
        story = await generate_story_from_grok(normalized_payload, request_id)
    except httpx.HTTPStatusError as exc:
        logger.exception("[%s] Story generation failed with upstream status error", request_id)
        raise HTTPException(status_code=502, detail="The story generation service returned an error.") from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Story generation failed while calling Grok", request_id)
        raise HTTPException(status_code=502, detail="Unable to reach the story generation service right now.") from exc
    except RuntimeError as exc:
        logger.exception("[%s] Story generation returned invalid data", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected story generation error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during story generation.") from exc

    return GenerateStoryResponse(
        status="success",
        story=story,
        story_type=story_type,
        duration=duration,
        language=language,
    )


@app.post("/generate-audio", response_model=GenerateAudioResponse)
async def generate_audio(request: Request, payload: GenerateAudioRequest) -> GenerateAudioResponse:
    request_id = uuid.uuid4().hex[:8]
    narration_text = normalize_text_input(payload.text)
    requested_voice = normalize_voice_value(payload.voice_model or STORY_TTS_DEFAULT_VOICE)

    if not narration_text:
        raise HTTPException(status_code=400, detail="Please generate a story before converting it to audio.")
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

    uvicorn.run("storynarration:app", host=STORY_APP_HOST, port=STORY_APP_PORT, reload=True)
