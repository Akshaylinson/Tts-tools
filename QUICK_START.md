# Quick Start Guide - Clean URLs

## Running with Clean URLs (Recommended)

The platform uses **FastAPI** for clean URL routing. This means URLs like `/audio-proofreader` work without `.html` extensions.

### Windows

Double-click `start-server.bat` or run:

```bash
cd "Adsense Tools Page"
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

### Access the Platform

Open your browser to:
- **Homepage:** http://127.0.0.1:8010/
- **Audio Proofreader:** http://127.0.0.1:8010/audio-proofreader
- **PDF to Audio:** http://127.0.0.1:8010/pdf-to-audio
- **Blog to Podcast:** http://127.0.0.1:8010/blog-to-podcast
- **Text to Audio:** http://127.0.0.1:8010/text-to-audio
- **Story Narration:** http://127.0.0.1:8010/story-narration-generator

## Important Notes

### ❌ Don't Use Live Server (Port 5500)
Live Server doesn't support URL rewriting, so clean URLs won't work. You'll get "Cannot GET /audio-proofreader" errors.

### ✅ Use FastAPI Server (Port 8010)
The FastAPI server in `app.py` handles all URL routing and API endpoints correctly.

### URL Structure
- ✅ **Clean URLs:** `/audio-proofreader` (works with FastAPI)
- ❌ **File paths:** `/tools/audio_proofreader/audio-proofreader.html` (only works with Live Server)

## Troubleshooting

**Problem:** "Cannot GET /audio-proofreader"
**Solution:** You're using Live Server. Stop it and run the FastAPI server instead.

**Problem:** Port 8010 is already in use
**Solution:** Change the port in the command:
```bash
uvicorn app:app --host 0.0.0.0 --port 8011 --reload
```

**Problem:** Module not found errors
**Solution:** Install dependencies:
```bash
pip install -r tools/pdf_to_audio/requirements.txt
pip install -r tools/Text-to-audio/requirements.txt
pip install -r tools/story_narration_generator/requirements.txt
pip install -r tools/audio_proofreader/requirements.txt
pip install -r tools/blog_to_podcast/requirements.txt
```
