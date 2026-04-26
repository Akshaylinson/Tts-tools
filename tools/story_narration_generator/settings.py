import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


STORY_APP_HOST = get_env("STORY_APP_HOST", "0.0.0.0")
STORY_APP_PORT = int(get_env("STORY_APP_PORT", "8010"))

STORY_GROK_BASE_URL = get_env("STORY_GROK_BASE_URL", "https://api.x.ai").rstrip("/")
STORY_GROK_API_KEY = get_env("STORY_GROK_API_KEY")
STORY_GROK_MODEL = get_env("STORY_GROK_MODEL", "grok-3-mini")
STORY_GROK_TIMEOUT_SECONDS = float(get_env("STORY_GROK_TIMEOUT_SECONDS", "60"))

STORY_TTS_BASE_URL = get_env("STORY_TTS_BASE_URL").rstrip("/")
STORY_TTS_API_KEY = get_env("STORY_TTS_API_KEY")
STORY_TTS_DEFAULT_VOICE = get_env("STORY_TTS_VOICE", "Ryan")
STORY_MAX_IDEA_CHARS = int(get_env("STORY_MAX_IDEA_CHARS", "1200"))
STORY_TTS_CHUNK_CHARS = int(get_env("STORY_TTS_CHUNK_CHARS", "1600"))
STORY_TTS_POLL_SECONDS = float(get_env("STORY_TTS_POLL_SECONDS", "15"))
STORY_TTS_POLL_ATTEMPTS = int(get_env("STORY_TTS_POLL_ATTEMPTS", "200"))


def validate_story_runtime_settings() -> None:
    if not STORY_GROK_API_KEY:
        raise RuntimeError(
            "Missing required environment variable: STORY_GROK_API_KEY. "
            "Set it in 'Adsense Tools Page/tools/story_narration_generator/.env' or your Docker environment."
        )


def validate_tts_runtime_settings() -> None:
    if not STORY_TTS_BASE_URL:
        raise RuntimeError(
            "Missing required environment variable: STORY_TTS_BASE_URL. "
            "Set it in 'Adsense Tools Page/tools/story_narration_generator/.env' or your Docker environment."
        )
