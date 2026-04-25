import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

load_dotenv(ROOT_DIR / ".env")
load_dotenv(APP_DIR / ".env", override=True)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


MAX_FILE_SIZE = int(get_env("PDF_AUDIO_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
MAX_TEXT_CHARS = int(get_env("PDF_AUDIO_MAX_TEXT_CHARS", "18000"))
TTS_CHUNK_CHARS = int(get_env("PDF_AUDIO_TTS_CHUNK_CHARS", "1800"))
TTS_POLL_SECONDS = float(get_env("PDF_AUDIO_TTS_POLL_SECONDS", "2"))
TTS_POLL_ATTEMPTS = int(get_env("PDF_AUDIO_TTS_POLL_ATTEMPTS", "90"))

APP_HOST = get_env("PDF_AUDIO_APP_HOST", "0.0.0.0")
APP_PORT = int(get_env("PDF_AUDIO_APP_PORT", "8010"))

TTS_BASE_URL = get_env("PDF_AUDIO_TTS_BASE_URL").rstrip("/")
TTS_API_KEY = get_env("PDF_AUDIO_TTS_API_KEY")
TTS_VOICE = get_env("PDF_AUDIO_TTS_VOICE", "ryan")

CLEANER_API_URL = get_env("PDF_AUDIO_CLEANER_API_URL")
CLEANER_API_KEY = get_env("PDF_AUDIO_CLEANER_API_KEY")
CLEANER_MODEL = get_env("PDF_AUDIO_CLEANER_MODEL", "gpt-4o-mini")


def validate_runtime_settings() -> None:
    if not TTS_BASE_URL:
        raise RuntimeError(
            "Missing required environment variable: PDF_AUDIO_TTS_BASE_URL. "
            "Set it in 'Adsense Tools Page/.env' or your Docker Compose environment."
        )
