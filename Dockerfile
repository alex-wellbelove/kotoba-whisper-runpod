FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install ffmpeg for yt-dlp audio extraction
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    torch \
    torchaudio \
    ffmpeg-python \
    yt-dlp \
    huggingface_hub --upgrade \
    filelock --upgrade

# Copy handler
COPY handler.py /app/handler.py

# Pre-download models during build (avoids runtime disk space issues)
RUN python -c "from transformers import pipeline; \
    pipeline('automatic-speech-recognition', 'kotoba-tech/kotoba-whisper-v2.1'); \
    pipeline('automatic-speech-recognition', 'openai/whisper-large-v3')"

CMD ["python", "-u", "/app/handler.py"]
