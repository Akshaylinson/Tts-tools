@echo off
echo Starting Piper Audio Platform...
echo.
echo Clean URLs will be available at:
echo   http://127.0.0.1:8010/
echo   http://127.0.0.1:8010/audio-proofreader
echo   http://127.0.0.1:8010/pdf-to-audio
echo   http://127.0.0.1:8010/blog-to-podcast
echo   http://127.0.0.1:8010/text-to-audio
echo   http://127.0.0.1:8010/story-narration-generator
echo.
echo Press Ctrl+C to stop the server
echo.
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
