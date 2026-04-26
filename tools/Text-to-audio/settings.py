import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


TEXT_AUDIO_MAX_CHARS = int(get_env("TEXT_AUDIO_MAX_CHARS", "5000"))
TEXT_AUDIO_TTS_CHUNK_CHARS = int(get_env("TEXT_AUDIO_TTS_CHUNK_CHARS", "1600"))
TEXT_AUDIO_TTS_POLL_SECONDS = float(get_env("TEXT_AUDIO_TTS_POLL_SECONDS", "15"))
TEXT_AUDIO_TTS_POLL_ATTEMPTS = int(get_env("TEXT_AUDIO_TTS_POLL_ATTEMPTS", "200"))

TEXT_AUDIO_APP_HOST = get_env("TEXT_AUDIO_APP_HOST", "0.0.0.0")
TEXT_AUDIO_APP_PORT = int(get_env("TEXT_AUDIO_APP_PORT", "8010"))

TEXT_AUDIO_TTS_BASE_URL = get_env("TEXT_AUDIO_TTS_BASE_URL").rstrip("/")
TEXT_AUDIO_TTS_API_KEY = get_env("TEXT_AUDIO_TTS_API_KEY")
TEXT_AUDIO_TTS_DEFAULT_VOICE = get_env("TEXT_AUDIO_TTS_VOICE", "Ryan")


def validate_runtime_settings() -> None:
    if not TEXT_AUDIO_TTS_BASE_URL:
        raise RuntimeError(
            "Missing required environment variable: TEXT_AUDIO_TTS_BASE_URL. "
            "Set it in 'Adsense Tools Page/tools/Text-to-audio/.env' or your Docker environment."
        )
