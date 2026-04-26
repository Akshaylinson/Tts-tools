FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY tools/Text-to-audio/requirements.txt ./tools/Text-to-audio/requirements.txt
RUN pip install --no-cache-dir -r ./tools/Text-to-audio/requirements.txt

COPY . .

EXPOSE 8010

CMD ["sh", "-c", "uvicorn app:app --host ${TEXT_AUDIO_APP_HOST:-0.0.0.0} --port ${TEXT_AUDIO_APP_PORT:-8010}"]
