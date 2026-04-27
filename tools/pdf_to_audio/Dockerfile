FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY tools/pdf_to_audio/requirements.txt ./tools/pdf_to_audio/requirements.txt
COPY tools/Text-to-audio/requirements.txt ./tools/Text-to-audio/requirements.txt
COPY tools/story_narration_generator/requirements.txt ./tools/story_narration_generator/requirements.txt
COPY tools/blog_to_podcast/requirements.txt ./tools/blog_to_podcast/requirements.txt
RUN pip install --no-cache-dir -r ./tools/pdf_to_audio/requirements.txt
RUN pip install --no-cache-dir -r ./tools/Text-to-audio/requirements.txt
RUN pip install --no-cache-dir -r ./tools/story_narration_generator/requirements.txt
RUN pip install --no-cache-dir -r ./tools/blog_to_podcast/requirements.txt

COPY . .

EXPOSE 8010

CMD ["sh", "-c", "uvicorn app:app --host ${PDF_AUDIO_APP_HOST:-0.0.0.0} --port ${PDF_AUDIO_APP_PORT:-8010}"]
