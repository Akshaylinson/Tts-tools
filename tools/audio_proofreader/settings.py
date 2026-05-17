import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


AUDIO_PROOFREADER_APP_HOST = get_env("AUDIO_PROOFREADER_APP_HOST", "0.0.0.0")
AUDIO_PROOFREADER_APP_PORT = int(get_env("AUDIO_PROOFREADER_APP_PORT", "8010"))

GROQ_API_KEY = get_env("GROQ_API_KEY")
GROQ_MODEL = get_env("GROQ_MODEL")
GROQ_API_URL = get_env("GROQ_API_URL").rstrip("/")
GROQ_TIMEOUT_SECONDS = float(get_env("GROQ_TIMEOUT_SECONDS", "60"))

CODEVOICE_API_KEY = get_env("CODEVOICE_API_KEY")
CODEVOICE_API_URL = get_env("CODEVOICE_API_URL").rstrip("/")
CODEVOICE_TIMEOUT_SECONDS = float(get_env("CODEVOICE_TIMEOUT_SECONDS", "180"))
CODEVOICE_VOICE_TIMEOUT_SECONDS = float(get_env("CODEVOICE_VOICE_TIMEOUT_SECONDS", "30"))

DEFAULT_LANGUAGE = get_env("DEFAULT_LANGUAGE")
DEFAULT_VOICE = get_env("DEFAULT_VOICE")

AUDIO_PROOFREADER_MAX_CHARS = int(get_env("AUDIO_PROOFREADER_MAX_CHARS", "10000"))
AUDIO_PROOFREADER_TTS_CHUNK_CHARS = int(get_env("AUDIO_PROOFREADER_TTS_CHUNK_CHARS", "1600"))
AUDIO_PROOFREADER_TTS_POLL_SECONDS = float(get_env("AUDIO_PROOFREADER_TTS_POLL_SECONDS", "15"))
AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS = int(get_env("AUDIO_PROOFREADER_TTS_POLL_ATTEMPTS", "200"))


def validate_analysis_runtime_settings() -> None:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GROQ_MODEL:
        missing.append("GROQ_MODEL")
    if not GROQ_API_URL:
        missing.append("GROQ_API_URL")
    if missing:
        raise RuntimeError(
            "Missing required environment variables for AI analysis: "
            + ", ".join(missing)
            + ". Set them in 'tools/audio_proofreader/.env' or your runtime environment."
        )


def validate_tts_runtime_settings() -> None:
    if not CODEVOICE_API_URL:
        raise RuntimeError(
            "Missing required environment variable: CODEVOICE_API_URL. "
            "Set it in 'tools/audio_proofreader/.env' or your runtime environment."
        )
