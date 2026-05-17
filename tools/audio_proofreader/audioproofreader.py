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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

try:
    from .settings import (
        APP_DIR,
        AUDIO_PROOFREADER_APP_HOST,
        AUDIO_PROOFREADER_APP_PORT,
        AUDIO_PROOFREADER_MAX_CHARS,
        AUDIO_PROOFREADER_TTS_CHUNK_CHARS,
        AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS,
        AUDIO_PROOFREADER_TTS_POLL_SECONDS,
        CODEVOICE_API_KEY,
        CODEVOICE_API_URL,
        CODEVOICE_TIMEOUT_SECONDS,
        CODEVOICE_VOICE_TIMEOUT_SECONDS,
        DEFAULT_LANGUAGE,
        DEFAULT_VOICE,
        GROQ_API_KEY,
        GROQ_API_URL,
        GROQ_MODEL,
        GROQ_TIMEOUT_SECONDS,
        validate_analysis_runtime_settings,
        validate_tts_runtime_settings,
    )
except ImportError:
    from settings import (
        APP_DIR,
        AUDIO_PROOFREADER_APP_HOST,
        AUDIO_PROOFREADER_APP_PORT,
        AUDIO_PROOFREADER_MAX_CHARS,
        AUDIO_PROOFREADER_TTS_CHUNK_CHARS,
        AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS,
        AUDIO_PROOFREADER_TTS_POLL_SECONDS,
        CODEVOICE_API_KEY,
        CODEVOICE_API_URL,
        CODEVOICE_TIMEOUT_SECONDS,
        CODEVOICE_VOICE_TIMEOUT_SECONDS,
        DEFAULT_LANGUAGE,
        DEFAULT_VOICE,
        GROQ_API_KEY,
        GROQ_API_URL,
        GROQ_MODEL,
        GROQ_TIMEOUT_SECONDS,
        validate_analysis_runtime_settings,
        validate_tts_runtime_settings,
    )


OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("audio-proofreader")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [audio-proofreader] %(message)s"))
    logger.addHandler(handler)

app = FastAPI(
    title="AI Audio Proofreader",
    description="Hear how your writing sounds before sending it with AI-guided spoken-flow feedback.",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

TOOL_ROUTE_PREFIX = "/audio-proofreader"
QUALITY_PRIORITY = {
    "ultra": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}
ALLOWED_SUBSCORES = {"low", "medium", "high"}


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


class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class SuggestionItem(BaseModel):
    original: str
    improved: str


class AnalysisSummary(BaseModel):
    confidence_score: int
    subscores: dict[str, str]
    flags: List[str]
    suggestions: List[SuggestionItem]


class AnalyzeTextResponse(AnalysisSummary):
    improved_text: str
    improved_analysis: Optional[AnalysisSummary] = None


class GenerateAudioRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: str = Field(..., min_length=1, max_length=100)
    speed: float = Field(default=1.0, ge=0.8, le=1.2)


class GenerateAudioResponse(BaseModel):
    audio_url: str
    filename: str
    voice: str
    speed: float


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
            (voice.language_name or voice.language or "Unknown").lower(),
            -voice.rating,
            -QUALITY_PRIORITY.get(voice.quality.lower(), 0),
            -voice.downloads,
            voice.label.lower(),
        ),
    )


async def fetch_available_tts_voices(client: httpx.AsyncClient) -> List[VoiceOption]:
    validate_tts_runtime_settings()
    headers = {}
    if CODEVOICE_API_KEY:
        headers["X-API-Key"] = CODEVOICE_API_KEY

    response = await client.get(f"{CODEVOICE_API_URL}/v1/voices", headers=headers)
    response.raise_for_status()
    payload = response.json()
    voices_payload = payload.get("voices")
    if not isinstance(voices_payload, list):
        raise RuntimeError("The voice service returned an invalid payload.")

    voices = [
        build_voice_option(voice)
        for voice in voices_payload
        if normalize_voice_value(str(voice.get("name") or voice.get("id") or ""))
    ]
    if not voices:
        raise RuntimeError("No voices are currently available.")
    return sort_voice_options(voices)


def get_default_voice(voices: List[VoiceOption]) -> str:
    configured = normalize_voice_value(DEFAULT_VOICE)
    preferred_language = (DEFAULT_LANGUAGE or "").strip().lower()
    if configured:
        for voice in voices:
            if voice.id.lower() == configured.lower() or voice.label.lower() == configured.lower():
                return voice.id
    if preferred_language:
        for voice in voices:
            labels = [voice.language.lower(), voice.language_name.lower()]
            if preferred_language in labels:
                return voice.id
    return voices[0].id if voices else configured


def get_fallback_voice_option() -> VoiceOption:
    fallback_voice = normalize_voice_value(DEFAULT_VOICE) or "Default Voice"
    fallback_language = (DEFAULT_LANGUAGE or "").strip() or "Unknown"
    return VoiceOption(
        id=fallback_voice,
        label=fallback_voice,
        language=fallback_language,
        language_name=fallback_language,
        gender="unknown",
        description="Fallback voice configured for the audio proofreader.",
        quality="unknown",
        rating=0,
        downloads=0,
    )


def chunk_text_for_tts(text: str) -> List[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        parts = [sentence]
        if len(sentence) > AUDIO_PROOFREADER_TTS_CHUNK_CHARS:
            parts = re.split(r"(?<=[,;:])\s+", sentence)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > AUDIO_PROOFREADER_TTS_CHUNK_CHARS:
                for index in range(0, len(part), AUDIO_PROOFREADER_TTS_CHUNK_CHARS):
                    segment = part[index:index + AUDIO_PROOFREADER_TTS_CHUNK_CHARS].strip()
                    if segment:
                        if current:
                            chunks.append(current)
                            current = ""
                        chunks.append(segment)
                continue

            tentative = f"{current} {part}".strip() if current else part
            if len(tentative) <= AUDIO_PROOFREADER_TTS_CHUNK_CHARS:
                current = tentative
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


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
            raise RuntimeError("The analysis model did not return a valid JSON object.")
        return json.loads(match.group(0))


def clean_subscore(value: str, label: str) -> str:
    cleaned = normalize_text_input(value).capitalize()
    if cleaned.lower() not in ALLOWED_SUBSCORES:
        raise RuntimeError(f"The analysis response returned an invalid {label} subscore.")
    return cleaned


def clean_flags(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = [normalize_text_input(str(item)) for item in values]
    return [item for item in cleaned if item][:5]


def clean_suggestions(values: object) -> List[SuggestionItem]:
    if not isinstance(values, list):
        return []

    suggestions: List[SuggestionItem] = []
    for item in values[:5]:
        if not isinstance(item, dict):
            continue
        original = normalize_text_input(str(item.get("original") or ""))
        improved = normalize_text_input(str(item.get("improved") or ""))
        if original and improved:
            suggestions.append(SuggestionItem(original=original, improved=improved))
    return suggestions


def clean_analysis_summary(payload: dict, *, require_suggestions: bool) -> AnalysisSummary:
    subscores = payload.get("subscores")
    if not isinstance(subscores, dict):
        raise RuntimeError("The analysis response is missing subscores.")

    confidence_score = int(payload.get("confidence_score"))
    if confidence_score < 0 or confidence_score > 100:
        raise RuntimeError("The analysis response returned an invalid confidence score.")

    suggestions = clean_suggestions(payload.get("suggestions"))
    if require_suggestions and not suggestions:
        raise RuntimeError("The analysis response did not include any actionable suggestions.")

    return AnalysisSummary(
        confidence_score=confidence_score,
        subscores={
            "clarity": clean_subscore(str(subscores.get("clarity") or ""), "clarity"),
            "professionalism": clean_subscore(str(subscores.get("professionalism") or ""), "professionalism"),
            "listening_flow": clean_subscore(str(subscores.get("listening_flow") or ""), "listening flow"),
        },
        flags=clean_flags(payload.get("flags")),
        suggestions=suggestions,
    )


def clean_analysis_response(payload: dict) -> AnalyzeTextResponse:
    improved_text = normalize_text_input(str(payload.get("improved_text") or ""))
    if not improved_text:
        raise RuntimeError("The analysis response is missing improved_text.")

    base_summary = clean_analysis_summary(payload, require_suggestions=True)
    improved_payload = payload.get("improved_analysis")
    improved_analysis = None
    if isinstance(improved_payload, dict):
        improved_analysis = clean_analysis_summary(improved_payload, require_suggestions=False)

    return AnalyzeTextResponse(
        confidence_score=base_summary.confidence_score,
        subscores=base_summary.subscores,
        flags=base_summary.flags,
        suggestions=base_summary.suggestions,
        improved_text=improved_text,
        improved_analysis=improved_analysis,
    )


def build_analysis_prompts(text: str) -> Tuple[str, str]:
    system_prompt = (
        "You are an AI audio proofreader. "
        "Analyze writing based on how it sounds when read aloud, not as a grammar teacher. "
        "Focus on spoken clarity, emotional tone, repetition, awkward phrasing, and listening rhythm. "
        "Keep rewrites natural, warm, and useful. Avoid robotic or over-formal phrasing. "
        "Return valid JSON only."
    )
    user_prompt = (
        "Review the text below as if the user wants to hear it before sending.\n\n"
        f"TEXT:\n{text}\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "confidence_score": integer 0-100,\n'
        '  "subscores": {"clarity":"Low|Medium|High","professionalism":"Low|Medium|High","listening_flow":"Low|Medium|High"},\n'
        '  "flags": ["short spoken-flow concern"],\n'
        '  "suggestions": [{"original":"exact or near-exact excerpt","improved":"better version for listening"}],\n'
        '  "improved_text": "full rewritten version",\n'
        '  "improved_analysis": {\n'
        '    "confidence_score": integer 0-100,\n'
        '    "subscores": {"clarity":"Low|Medium|High","professionalism":"Low|Medium|High","listening_flow":"Low|Medium|High"},\n'
        '    "flags": ["remaining concern if any"],\n'
        '    "suggestions": []\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Treat the confidence score as a practical heuristic, not a scientific metric.\n"
        "- Keep flags short, specific, and easy to scan.\n"
        "- Include 2 to 5 suggestions when there are issues.\n"
        "- Keep the improved text close to the user's intent.\n"
        "- Optimize for spoken rhythm and emotional tone.\n"
        "- Do not add greetings, explanations, markdown, or code fences.\n"
    )
    return system_prompt, user_prompt


async def analyze_text_with_groq(text: str, request_id: str) -> AnalyzeTextResponse:
    validate_analysis_runtime_settings()
    system_prompt, user_prompt = build_analysis_prompts(text)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{GROQ_API_URL}/chat/completions", headers=headers, json=body)
        if response.is_error:
            logger.error(
                "[%s] Groq upstream error %s from %s: %s",
                request_id,
                response.status_code,
                response.request.url,
                response.text,
            )
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("The analysis model returned no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The analysis model returned an empty response.")

    logger.info("[%s] Audio proofreading analysis received", request_id)
    return clean_analysis_response(extract_json_object(content))


async def create_tts_job(client: httpx.AsyncClient, text: str, voice: str, speed: float) -> str:
    validate_tts_runtime_settings()
    headers = {"Content-Type": "application/json"}
    if CODEVOICE_API_KEY:
        headers["X-API-Key"] = CODEVOICE_API_KEY

    response = await client.post(
        f"{CODEVOICE_API_URL}/v1/tts",
        headers=headers,
        json={"text": text, "voice": voice, "speed": speed},
    )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError("No job_id returned from the voice service.")
    return job_id


async def wait_for_tts_audio(client: httpx.AsyncClient, job_id: str, request_id: str, chunk_index: int) -> Tuple[bytes, str]:
    headers = {}
    if CODEVOICE_API_KEY:
        headers["X-API-Key"] = CODEVOICE_API_KEY

    for attempt in range(1, AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(AUDIO_PROOFREADER_TTS_POLL_SECONDS)
        status_response = await client.get(f"{CODEVOICE_API_URL}/tts/status/{job_id}", headers=headers)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        logger.info(
            "[%s] Chunk %s poll %s/%s for job %s returned status=%s",
            request_id,
            chunk_index,
            attempt,
            AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS,
            job_id,
            status,
        )

        if status == "completed":
            audio_format = str(status_payload.get("audio_format") or "MP3").lower()
            audio_response = await client.get(f"{CODEVOICE_API_URL}/v1/audio/{job_id}", headers=headers)
            audio_response.raise_for_status()
            return audio_response.content, audio_format

        if status == "failed":
            raise RuntimeError(status_payload.get("error") or "The voice preview request failed.")

    raise RuntimeError("The voice preview timed out before audio was ready.")


async def convert_text_to_audio(clean_text: str, request_id: str, voice: str, speed: float) -> str:
    chunks = chunk_text_for_tts(clean_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The text is empty after cleanup.")

    merged_audio: Optional[AudioSegment] = None
    logger.info("[%s] Split preview text into %s chunk(s)", request_id, len(chunks))

    async with httpx.AsyncClient(timeout=CODEVOICE_TIMEOUT_SECONDS) as client:
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "[%s] Sending chunk %s/%s to CodeVoice (%s chars). Preview: \"%s\"",
                request_id,
                chunk_index,
                len(chunks),
                len(chunk),
                make_log_preview(chunk),
            )
            job_id = await create_tts_job(client, chunk, voice, speed)
            audio_bytes, audio_format = await wait_for_tts_audio(client, job_id, request_id, chunk_index)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav" if audio_format == "wav" else "mp3")
            merged_audio = segment if merged_audio is None else merged_audio + segment

    stem_source = re.sub(r"[^a-zA-Z0-9_-]+", "-", make_log_preview(clean_text, limit=36)).strip("-").lower() or "audio-proofreader"
    output_name = f"{stem_source}-{uuid.uuid4().hex[:10]}.mp3"
    output_path = OUTPUTS_DIR / output_name
    merged_audio.export(output_path, format="mp3", bitrate="128k")
    return output_name


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def audio_proofreader_page() -> HTMLResponse:
    return HTMLResponse(read_html_file("audio-proofreader.html"))


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/voices", response_model=VoicesResponse)
async def list_tts_voices() -> VoicesResponse:
    try:
        async with httpx.AsyncClient(timeout=CODEVOICE_VOICE_TIMEOUT_SECONDS) as client:
            voices = await fetch_available_tts_voices(client)
        return VoicesResponse(voices=voices, default_voice=get_default_voice(voices))
    except Exception as exc:
        logger.warning("Unable to load live CodeVoice voices (%s). Falling back to configured default.", exc)
        fallback = get_fallback_voice_option()
        return VoicesResponse(voices=[fallback], default_voice=fallback.id)


@app.post("/analyze-text", response_model=AnalyzeTextResponse)
async def analyze_text(payload: AnalyzeTextRequest) -> AnalyzeTextResponse:
    request_id = uuid.uuid4().hex[:8]
    text = normalize_text_input(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="Please paste some writing before analyzing it.")
    if len(text) > AUDIO_PROOFREADER_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long. The current limit is {AUDIO_PROOFREADER_MAX_CHARS:,} characters.",
        )

    try:
        logger.info("[%s] Analyzing %s characters for spoken-flow feedback", request_id, len(text))
        return await analyze_text_with_groq(text, request_id)
    except httpx.HTTPStatusError as exc:
        logger.exception("[%s] Analysis failed with upstream status error", request_id)
        upstream_detail = exc.response.text.strip() if exc.response is not None and exc.response.text else ""
        detail = "The writing analysis service returned an error."
        if upstream_detail:
            detail = f"{detail} Upstream response: {upstream_detail}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Analysis failed while calling Groq", request_id)
        raise HTTPException(status_code=502, detail="Unable to reach the writing analysis service right now.") from exc
    except RuntimeError as exc:
        logger.exception("[%s] Analysis returned invalid data", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected analysis error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during analysis.") from exc


@app.post("/generate-audio", response_model=GenerateAudioResponse)
async def generate_audio(request: Request, payload: GenerateAudioRequest) -> GenerateAudioResponse:
    request_id = uuid.uuid4().hex[:8]
    text = normalize_text_input(payload.text)
    requested_voice = normalize_voice_value(payload.voice)
    speed = round(float(payload.speed), 2)

    if not text:
        raise HTTPException(status_code=400, detail="Please paste some writing before generating audio.")
    if len(text) > AUDIO_PROOFREADER_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long. The current limit is {AUDIO_PROOFREADER_MAX_CHARS:,} characters.",
        )
    if speed not in {0.9, 1.0, 1.1}:
        raise HTTPException(status_code=400, detail="Please choose a supported playback speed.")
    if not requested_voice:
        raise HTTPException(status_code=400, detail="Please choose a voice before generating audio.")

    try:
        try:
            async with httpx.AsyncClient(timeout=CODEVOICE_VOICE_TIMEOUT_SECONDS) as client:
                available_voices = await fetch_available_tts_voices(client)
            matching_voice = next(
                (
                    item
                    for item in available_voices
                    if item.id.lower() == requested_voice.lower() or item.label.lower() == requested_voice.lower()
                ),
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

        output_name = await convert_text_to_audio(text, request_id, selected_voice, speed)
        output_url = str(request.base_url).rstrip("/") + f"{TOOL_ROUTE_PREFIX}/outputs/{output_name}"
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("[%s] Audio generation failed during TTS processing", request_id)
        detail = str(exc)
        status_code = 504 if "timed out" in detail.lower() else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Audio generation failed while calling CodeVoice", request_id)
        raise HTTPException(status_code=502, detail="Failed to reach the configured voice service.") from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected audio generation error", request_id)
        raise HTTPException(status_code=500, detail="Unexpected server error during audio generation.") from exc

    return GenerateAudioResponse(audio_url=output_url, filename=output_name, voice=selected_voice, speed=speed)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("audioproofreader:app", host=AUDIO_PROOFREADER_APP_HOST, port=AUDIO_PROOFREADER_APP_PORT, reload=True)
