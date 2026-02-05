"""
RunPod Serverless Handler for Kotoba-Whisper v2.0
Japanese transcription optimized for gaming/YouTube content
"""

import runpod
import torch
import base64
import tempfile
import os
import re
import subprocess
import urllib.request
from transformers import pipeline


# Load model at cold start (cached for warm starts)
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.0",
        torch_dtype=torch.float16,
        device="cuda",
    )


MODEL = load_model()

# Pattern to detect YouTube URLs
YOUTUBE_PATTERN = re.compile(
    r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
)


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_PATTERN.match(url))


def download_youtube_audio(url: str, output_path: str) -> dict:
    """Download audio from YouTube using yt-dlp. Returns video metadata."""
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_path.replace(".mp3", ".%(ext)s"),
        "--print-json",  # Output metadata
        "--no-playlist",  # Single video only
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    metadata = json.loads(result.stdout.strip().split('\n')[-1])
    return metadata


def transcribe(audio_path, return_timestamps=True, chunk_length_s=30, batch_size=16):
    """Run transcription with chunking for long audio."""
    result = MODEL(
        audio_path,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=return_timestamps,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )
    return result


def handler(job):
    """
    RunPod serverless handler.

    Input formats:
        - {"audio_base64": "<base64 encoded audio>"}
        - {"audio_url": "https://..."}  (direct audio URL)
        - {"youtube_url": "https://youtube.com/watch?v=..."}

    Optional params:
        - return_timestamps: bool (default: True)
        - chunk_length_s: int (default: 30)
        - batch_size: int (default: 16)
    """
    job_input = job["input"]

    # Get optional params
    return_timestamps = job_input.get("return_timestamps", True)
    chunk_length_s = job_input.get("chunk_length_s", 30)
    batch_size = job_input.get("batch_size", 16)

    metadata = None

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Handle base64 input
        if "audio_base64" in job_input:
            audio_data = base64.b64decode(job_input["audio_base64"])
            with open(audio_path, "wb") as f:
                f.write(audio_data)

        # Handle YouTube URL
        elif "youtube_url" in job_input:
            url = job_input["youtube_url"]
            metadata = download_youtube_audio(url, audio_path)

        # Handle direct audio URL
        elif "audio_url" in job_input:
            url = job_input["audio_url"]
            # Auto-detect YouTube URLs passed to audio_url
            if is_youtube_url(url):
                metadata = download_youtube_audio(url, audio_path)
            else:
                urllib.request.urlretrieve(url, audio_path)

        else:
            return {"error": "Must provide 'audio_base64', 'audio_url', or 'youtube_url'"}

        # Transcribe
        result = transcribe(
            audio_path,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
        )

    output = {
        "text": result["text"],
        "chunks": result.get("chunks", []),
    }

    # Include video metadata if available
    if metadata:
        output["title"] = metadata.get("title")
        output["duration"] = metadata.get("duration")
        output["channel"] = metadata.get("channel")

    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
