FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    torch \
    torchaudio \
    ffmpeg-python \
    yt-dlp

# Install flash attention for faster inference
RUN pip install flash-attn --no-build-isolation

# Pre-download the model during build (faster cold starts)
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='kotoba-tech/kotoba-whisper-v2.0')"

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
