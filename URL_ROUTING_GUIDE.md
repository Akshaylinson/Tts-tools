# URL Routing - Live Server vs FastAPI

## The Problem You Encountered

When using **Live Server (Port 5500)**, you saw:
```
Cannot GET /audio-proofreader
```

This happens because Live Server is a simple static file server that doesn't understand URL routing.

## The Solution

Use the **FastAPI Server (Port 8010)** which has proper URL routing configured.

---

## Comparison

### ❌ Live Server (Port 5500) - Static Files Only

**What works:**
- `http://127.0.0.1:5500/index.html`
- `http://127.0.0.1:5500/tools/audio_proofreader/audio-proofreader.html`
- `http://127.0.0.1:5500/tools/pdf_to_audio/pdf-to-audio.html`

**What doesn't work:**
- `http://127.0.0.1:5500/audio-proofreader` ❌ Cannot GET
- `http://127.0.0.1:5500/pdf-to-audio` ❌ Cannot GET

**Why:** Live Server only serves files at their exact file paths. It doesn't do URL rewriting.

---

### ✅ FastAPI Server (Port 8010) - Clean URLs

**What works:**
- `http://127.0.0.1:8010/` → Homepage
- `http://127.0.0.1:8010/audio-proofreader` → Audio Proofreader tool
- `http://127.0.0.1:8010/pdf-to-audio` → PDF to Audio tool
- `http://127.0.0.1:8010/blog-to-podcast` → Blog to Podcast tool
- `http://127.0.0.1:8010/text-to-audio` → Text to Audio tool
- `http://127.0.0.1:8010/story-narration-generator` → Story Narration tool

**Why:** FastAPI's `app.mount()` handles URL routing and serves the correct tool at clean URLs.

---

## How to Start FastAPI Server

### Option 1: Double-click the batch file
```
start-server.bat
```

### Option 2: Run manually
```bash
cd "e:\AI_INFLUC_ SAAS\PIPER\piper-test\Adsense Tools Page"
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

### Option 3: VS Code Terminal
1. Open terminal in VS Code (Ctrl + `)
2. Navigate to project folder
3. Run: `uvicorn app:app --host 0.0.0.0 --port 8010 --reload`

---

## What Changed in Your Code?

**Nothing!** The clean URLs were already configured in `app.py`:

```python
# These lines in app.py create the clean URLs:
app.mount("/pdf-to-audio", pdf_to_audio_app)
app.mount("/text-to-audio", text_to_audio_app)
app.mount("/story-narration-generator", story_narration_app)
app.mount("/blog-to-podcast", blog_to_podcast_app)
app.mount("/audio-proofreader", audio_proofreader_app)
```

You just needed to use the **right server** (FastAPI instead of Live Server).

---

## Summary

| Feature | Live Server | FastAPI Server |
|---------|-------------|----------------|
| Port | 5500 | 8010 |
| Clean URLs | ❌ No | ✅ Yes |
| API Endpoints | ❌ No | ✅ Yes |
| URL Routing | ❌ No | ✅ Yes |
| File Changes Auto-reload | ✅ Yes | ✅ Yes |
| Best for | Static HTML preview | Full web applications |

**For Piper Audio, always use FastAPI Server (Port 8010).**
