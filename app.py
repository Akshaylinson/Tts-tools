import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from tools.pdf_to_audio.pdftoaudio import app as pdf_to_audio_app


APP_DIR = Path(__file__).resolve().parent


def load_tool_app(package_dir: Path, module_name: str):
    package_init = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_init,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool package from {package_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.app


text_to_audio_app = load_tool_app(APP_DIR / "tools" / "Text-to-audio", "text_to_audio_tool")
story_narration_app = load_tool_app(APP_DIR / "tools" / "story_narration_generator", "story_narration_tool")


def read_html_file(name: str) -> str:
    path = APP_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return path.read_text(encoding="utf-8")


app = FastAPI(
    title="Adsense Tools",
    description="Shared landing pages plus mounted tool apps.",
)
app.mount("/assets", StaticFiles(directory=str(APP_DIR / "assets")), name="assets")


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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> RedirectResponse:
    return RedirectResponse(url="/assets/favicon.svg", status_code=307)


app.mount("/pdf-to-audio", pdf_to_audio_app)
app.mount("/text-to-audio", text_to_audio_app)
app.mount("/story-narration-generator", story_narration_app)
