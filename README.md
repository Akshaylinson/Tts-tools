# Adsense Tools Page

This project is now organized for multiple tools.

## Structure

```text
Adsense Tools Page/
|-- app.py
|-- docker-compose.yml
|-- index.html
|-- about.html
|-- privacy-policy.html
|-- terms.html
|-- tools/
|   `-- pdf_to_audio/
|       |-- .env
|       |-- .env.example
|       |-- Dockerfile
|       |-- pdf-to-audio.html
|       |-- pdftoaudio.py
|       |-- requirements.txt
|       |-- settings.py
|       `-- outputs/
```

## How It Works

- `app.py` is the shared root app for common pages.
- `tools/pdf_to_audio/` contains all PDF-to-audio-specific code, config, HTML, Docker build files, and outputs.
- The root `docker-compose.yml` builds the whole site using the PDF tool Dockerfile and serves the shared app entrypoint.

## Run Locally

```bash
cd "Adsense Tools Page"
pip install -r tools/pdf_to_audio/requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

Open `http://127.0.0.1:8010/`.

## Run With Docker

```bash
cd "Adsense Tools Page"
docker compose up --build
```

The compose file reads PDF tool settings from `tools/pdf_to_audio/.env`.

## Adding More Tools

Follow the same pattern:

- create a folder inside `tools/`
- keep that tool's Python, HTML, env, requirements, Docker assets, and outputs inside its own folder
- mount or register the tool from `app.py`
