from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from tools.pdf_to_audio.pdftoaudio import app as pdf_to_audio_app


APP_DIR = Path(__file__).resolve().parent


def read_html_file(name: str) -> str:
    path = APP_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return path.read_text(encoding="utf-8")


app = FastAPI(
    title="Adsense Tools",
    description="Shared landing pages plus mounted tool apps.",
)


@app.get("/", include_in_schema=False)
async def home() -> HTMLResponse:
    return HTMLResponse(read_html_file("index.html"))


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


app.mount("/", pdf_to_audio_app)
