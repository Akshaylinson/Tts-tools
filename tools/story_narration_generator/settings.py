import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_env_alias(names: list[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


STORY_APP_HOST = get_env("STORY_APP_HOST", "0.0.0.0")
STORY_APP_PORT = int(get_env("STORY_APP_PORT", "8010"))

STORY_GROQ_BASE_URL = get_env_alias(
    ["STORY_GROQ_BASE_URL", "STORY_GROK_BASE_URL"],
    "https://api.groq.com/openai/v1",
).rstrip("/")
STORY_GROQ_API_KEY = get_env_alias(["STORY_GROQ_API_KEY", "STORY_GROK_API_KEY"])
STORY_GROQ_MODEL = get_env_alias(
    ["STORY_GROQ_MODEL", "STORY_GROK_MODEL"],
    "llama-3.3-70b-versatile",
)
STORY_GROQ_TIMEOUT_SECONDS = float(
    get_env_alias(["STORY_GROQ_TIMEOUT_SECONDS", "STORY_GROK_TIMEOUT_SECONDS"], "60")
)

STORY_TTS_BASE_URL = get_env("STORY_TTS_BASE_URL").rstrip("/")
STORY_TTS_API_KEY = get_env("STORY_TTS_API_KEY")
STORY_TTS_DEFAULT_VOICE = get_env("STORY_TTS_VOICE", "Ryan")
STORY_MAX_IDEA_CHARS = int(get_env("STORY_MAX_IDEA_CHARS", "1200"))
STORY_TTS_CHUNK_CHARS = int(get_env("STORY_TTS_CHUNK_CHARS", "1600"))
STORY_TTS_POLL_SECONDS = float(get_env("STORY_TTS_POLL_SECONDS", "15"))
STORY_TTS_POLL_ATTEMPTS = int(get_env("STORY_TTS_POLL_ATTEMPTS", "200"))


def validate_story_runtime_settings() -> None:
    if not STORY_GROQ_API_KEY:
        raise RuntimeError(
            "Missing required environment variable: STORY_GROQ_API_KEY. "
            "Set it in 'Adsense Tools Page/tools/story_narration_generator/.env' or your Docker environment."
        )


def validate_tts_runtime_settings() -> None:
    if not STORY_TTS_BASE_URL:
        raise RuntimeError(
            "Missing required environment variable: STORY_TTS_BASE_URL. "
            "Set it in 'Adsense Tools Page/tools/story_narration_generator/.env' or your Docker environment."
        )
