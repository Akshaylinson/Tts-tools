FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY tools/story_narration_generator/requirements.txt ./tools/story_narration_generator/requirements.txt
RUN pip install --no-cache-dir -r ./tools/story_narration_generator/requirements.txt

COPY . .

EXPOSE 8010

CMD ["sh", "-c", "uvicorn app:app --host ${STORY_APP_HOST:-0.0.0.0} --port ${STORY_APP_PORT:-8010}"]
