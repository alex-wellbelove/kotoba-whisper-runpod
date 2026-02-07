"""
RunPod Serverless Handler for Kotoba-Whisper v2.1
Japanese transcription optimized for gaming/YouTube content

Changes from v2.0:
- Updated to kotoba-whisper-v2.1 (better punctuation/timestamps via stable-ts)
- Use bfloat16 instead of float16 (recommended by HuggingFace)
- Added SDPA attention implementation for speed
- Default to sequential long-form mode (more accurate, no chunk_length_s)
- Optional chunked mode with chunk_length_s=15 (faster, use mode="fast")
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


# Global model cache
MODEL = None
CURRENT_MODEL_NAME = None

# Available models
MODELS = {
    "kotoba": "kotoba-tech/kotoba-whisper-v2.1",
    "large-v3": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
}
DEFAULT_MODEL = "kotoba"


def get_model(model_name: str = None):
    """Load model lazily on first request. Supports model switching."""
    global MODEL, CURRENT_MODEL_NAME
    
    model_name = model_name or DEFAULT_MODEL
    model_id = MODELS.get(model_name, model_name)  # Allow direct HF model IDs too
    
    if MODEL is None or CURRENT_MODEL_NAME != model_id:
        # Clear old model from GPU memory
        if MODEL is not None:
            del MODEL
            torch.cuda.empty_cache()
        
        MODEL = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        CURRENT_MODEL_NAME = model_id
    return MODEL


# Pattern to detect YouTube URLs
YOUTUBE_PATTERN = re.compile(
    r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
)


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_PATTERN.match(url))


def download_youtube_audio(url: str, output_path: str) -> dict:
    """Download audio from YouTube using yt-dlp. Returns video metadata."""
    import json
    
    # Update yt-dlp first (YouTube changes frequently)
    subprocess.run(["pip", "install", "-U", "yt-dlp"], capture_output=True)
    
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    metadata = json.loads(result.stdout.strip().split('\n')[-1])
    return metadata


def transcribe(audio_path, return_timestamps=True, mode="accurate", batch_size=16, model_name=None):
    """
    Run transcription on audio file.
    
    Args:
        audio_path: Path to audio file
        return_timestamps: Whether to return word/chunk timestamps
        mode: "accurate" (sequential, slower) or "fast" (chunked, faster)
        batch_size: Batch size for chunked mode
        model_name: Model to use ("kotoba", "large-v3", "large-v2", or HF model ID)
    
    Notes:
        - "accurate" mode: Uses sequential long-form algorithm (no chunking)
          Better for transcription quality, recommended for most cases.
        - "fast" mode: Uses chunked algorithm with 15s chunks
          Up to 9x faster but may have slight accuracy loss at chunk boundaries.
    """
    model = get_model(model_name)
    
    generate_kwargs = {"language": "ja", "task": "transcribe"}
    
    if mode == "fast":
        # Chunked mode: faster but experimental
        # chunk_length_s=15 is optimal for distil-whisper architecture
        result = model(
            audio_path,
            chunk_length_s=15,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs,
        )
    else:
        # Sequential mode: more accurate (default)
        # No chunk_length_s = uses Whisper's native long-form algorithm
        result = model(
            audio_path,
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs,
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
        - mode: "accurate" or "fast" (default: "accurate")
        - batch_size: int (default: 16, only used in fast mode)
        - model: "kotoba", "large-v3", "large-v2", or HF model ID (default: "kotoba")
    
    Deprecated params (for backwards compatibility):
        - chunk_length_s: If provided, enables fast mode
    """
    job_input = job["input"]

    # Get optional params
    return_timestamps = job_input.get("return_timestamps", True)
    batch_size = job_input.get("batch_size", 16)
    model_name = job_input.get("model", DEFAULT_MODEL)
    
    # Mode selection: explicit mode param, or infer from chunk_length_s
    mode = job_input.get("mode", "accurate")
    if "chunk_length_s" in job_input:
        # Backwards compatibility: if chunk_length_s is provided, use fast mode
        mode = "fast"

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
            mode=mode,
            batch_size=batch_size,
            model_name=model_name,
        )

    output = {
        "text": result["text"],
        "chunks": result.get("chunks", []),
        "mode": mode,
        "model": model_name,
    }

    # Include video metadata if available
    if metadata:
        output["title"] = metadata.get("title")
        output["duration"] = metadata.get("duration")
        output["channel"] = metadata.get("channel")

    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
