FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY tools/audio_proofreader/requirements.txt ./tools/audio_proofreader/requirements.txt
RUN pip install --no-cache-dir -r ./tools/audio_proofreader/requirements.txt

COPY . .

EXPOSE 8010

CMD ["sh", "-c", "uvicorn app:app --host ${AUDIO_PROOFREADER_APP_HOST:-0.0.0.0} --port ${AUDIO_PROOFREADER_APP_PORT:-8010}"]
