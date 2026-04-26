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
|   |-- pdf_to_audio/
|   |   |-- .env
|   |   |-- .env.example
|   |   |-- Dockerfile
|   |   |-- pdf-to-audio.html
|   |   |-- pdftoaudio.py
|   |   |-- requirements.txt
|   |   |-- settings.py
|   |   `-- outputs/
|   `-- Text-to-audio/
|       |-- .env
|       |-- .env.example
|       |-- Dockerfile
|       |-- text-to-audio.html
|       |-- texttoaudio.py
|       |-- requirements.txt
|       |-- settings.py
|       `-- outputs/
|   `-- story_narration_generator/
|       |-- .env
|       |-- .env.example
|       |-- Dockerfile
|       |-- story-narration-generator.html
|       |-- storynarration.py
|       |-- requirements.txt
|       |-- settings.py
|       `-- outputs/
```

## How It Works

- `app.py` is the shared root app for common pages.
- `tools/pdf_to_audio/` contains all PDF-to-audio-specific code, config, HTML, Docker build files, and outputs.
- `tools/Text-to-audio/` contains all text-to-audio-specific code, config, HTML, Docker build files, and outputs.
- `tools/story_narration_generator/` contains the AI story narration generator page, Groq prompt logic, TTS integration, HTML, Docker assets, and outputs.
- The root `docker-compose.yml` builds the whole site using the PDF tool Dockerfile and serves the shared app entrypoint.

## Run Locally

```bash
cd "Adsense Tools Page"
pip install -r tools/pdf_to_audio/requirements.txt
pip install -r tools/Text-to-audio/requirements.txt
pip install -r tools/story_narration_generator/requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

Open `http://127.0.0.1:8010/`.

## Run With Docker

```bash
cd "Adsense Tools Page"
docker compose up --build
```

The compose file currently reads runtime settings from:

- `tools/pdf_to_audio/.env`
- `tools/Text-to-audio/.env`
- `tools/story_narration_generator/.env`

## Adding More Tools

Follow the same pattern:

- create a folder inside `tools/`
- keep that tool's Python, HTML, env, requirements, Docker assets, and outputs inside its own folder
- mount or register the tool from `app.py`
