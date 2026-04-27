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


BLOG_PODCAST_APP_HOST = get_env("BLOG_PODCAST_APP_HOST", "0.0.0.0")
BLOG_PODCAST_APP_PORT = int(get_env("BLOG_PODCAST_APP_PORT", "8010"))

BLOG_GROK_BASE_URL = get_env_alias(
    ["BLOG_GROQ_BASE_URL", "BLOG_GROK_BASE_URL", "STORY_GROQ_BASE_URL", "STORY_GROK_BASE_URL"],
    "https://api.groq.com/openai/v1",
).rstrip("/")
BLOG_GROK_API_KEY = get_env_alias(
    ["BLOG_GROQ_API_KEY", "BLOG_GROK_API_KEY", "STORY_GROQ_API_KEY", "STORY_GROK_API_KEY"]
)
BLOG_GROK_MODEL = get_env_alias(
    ["BLOG_GROQ_MODEL", "BLOG_GROK_MODEL", "STORY_GROQ_MODEL", "STORY_GROK_MODEL"],
    "llama-3.3-70b-versatile",
)
BLOG_GROK_TIMEOUT_SECONDS = float(
    get_env_alias(
        ["BLOG_GROQ_TIMEOUT_SECONDS", "BLOG_GROK_TIMEOUT_SECONDS", "STORY_GROQ_TIMEOUT_SECONDS", "STORY_GROK_TIMEOUT_SECONDS"],
        "60",
    )
)

BLOG_TTS_BASE_URL = get_env("BLOG_TTS_BASE_URL").rstrip("/")
BLOG_TTS_API_KEY = get_env("BLOG_TTS_API_KEY")
BLOG_TTS_DEFAULT_VOICE = get_env("BLOG_TTS_VOICE", "Ryan")

BLOG_MAX_INPUT_CHARS = int(get_env("BLOG_MAX_INPUT_CHARS", "18000"))
BLOG_TTS_CHUNK_CHARS = int(get_env("BLOG_TTS_CHUNK_CHARS", "1600"))
BLOG_TTS_POLL_SECONDS = float(get_env("BLOG_TTS_POLL_SECONDS", "15"))
BLOG_TTS_POLL_ATTEMPTS = int(get_env("BLOG_TTS_POLL_ATTEMPTS", "200"))
BLOG_FETCH_TIMEOUT_SECONDS = float(get_env("BLOG_FETCH_TIMEOUT_SECONDS", "25"))
BLOG_FETCH_MAX_CHARS = int(get_env("BLOG_FETCH_MAX_CHARS", "24000"))


def validate_grok_runtime_settings() -> None:
    if not BLOG_GROK_API_KEY:
        raise RuntimeError(
            "Missing required Groq API configuration. "
            "Set BLOG_GROQ_API_KEY in 'Adsense Tools Page/tools/blog_to_podcast/.env', "
            "or provide STORY_GROQ_API_KEY for shared use."
        )


def validate_tts_runtime_settings() -> None:
    if not BLOG_TTS_BASE_URL:
        raise RuntimeError(
            "Missing required environment variable: BLOG_TTS_BASE_URL. "
            "Set it in 'Adsense Tools Page/tools/blog_to_podcast/.env' or your Docker environment."
        )
