#!/usr/bin/env python3
"""
Client script for Kotoba-Whisper RunPod serverless endpoint.

Usage:
    ./transcribe.py <audio_file_or_url> [--output srt|json|txt]

Examples:
    ./transcribe.py video.mp4 > output.srt
    ./transcribe.py "https://youtube.com/watch?v=xxxxx" > output.srt
    ./transcribe.py "https://youtu.be/xxxxx" --output json

Requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import requests


YOUTUBE_PATTERN = re.compile(
    r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
)


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_PATTERN.match(url))


def get_config():
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if not endpoint_id or not api_key:
        print("Error: Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        sys.exit(1)

    return endpoint_id, api_key


def submit_job_file(audio_path: str, endpoint_id: str, api_key: str, mode: str = "fast", model: str = "kotoba") -> str:
    """Submit transcription job with local file."""
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "input": {
                "audio_base64": audio_base64,
                "return_timestamps": True,
                "mode": mode,
                "model": model,
            }
        },
    )
    response.raise_for_status()
    return response.json()["id"]


def submit_job_url(url: str, endpoint_id: str, api_key: str, mode: str = "fast", model: str = "kotoba") -> str:
    """Submit transcription job with URL (YouTube or direct audio)."""
    input_data = {
        "return_timestamps": True,
        "mode": mode,
        "model": model,
    }

    if is_youtube_url(url):
        input_data["youtube_url"] = url
    else:
        input_data["audio_url"] = url

    response = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"input": input_data},
    )
    response.raise_for_status()
    return response.json()["id"]


def poll_status(job_id: str, endpoint_id: str, api_key: str) -> dict:
    """Poll for job completion."""
    while True:
        response = requests.get(
            f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()

        status = data["status"]
        if status == "COMPLETED":
            return data["output"]
        elif status == "FAILED":
            print(f"Job failed: {data.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)

        print(f"Status: {status}...", file=sys.stderr)
        time.sleep(2)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    if seconds is None:
        return "99:59:59,999"  # Fallback for missing end timestamps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(chunks: list) -> str:
    """Convert chunks to SRT format."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        ts = chunk.get("timestamp", [None, None])
        start = format_timestamp(ts[0] if ts else None)
        end = format_timestamp(ts[1] if ts and len(ts) > 1 else None)
        text = chunk.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Kotoba-Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s video.mp4 > output.srt
    %(prog)s "https://youtube.com/watch?v=xxxxx" -o srt > video.srt
    %(prog)s "https://youtu.be/xxxxx" --output json
        """,
    )
    parser.add_argument(
        "input",
        help="Path to audio/video file OR YouTube URL",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["srt", "json", "txt"],
        default="srt",
        help="Output format (default: srt)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["fast", "accurate"],
        default="fast",
        help="Transcription mode: 'fast' (chunked, GPU batching) or 'accurate' (sequential). Default: fast",
    )
    parser.add_argument(
        "--model",
        choices=["kotoba", "large-v3", "large-v2"],
        default="kotoba",
        help="Model: 'kotoba' (Japanese-optimized), 'large-v3', 'large-v2' (OpenAI Whisper). Default: kotoba",
    )
    args = parser.parse_args()

    endpoint_id, api_key = get_config()

    if is_url(args.input):
        print(f"Submitting URL: {args.input} (mode={args.mode}, model={args.model})", file=sys.stderr)
        job_id = submit_job_url(args.input, endpoint_id, api_key, mode=args.mode, model=args.model)
    else:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"Submitting file: {args.input} (mode={args.mode}, model={args.model})", file=sys.stderr)
        job_id = submit_job_file(args.input, endpoint_id, api_key, mode=args.mode, model=args.model)

    print(f"Job ID: {job_id}", file=sys.stderr)

    result = poll_status(job_id, endpoint_id, api_key)

    # Show video info if available
    if result.get("title"):
        print(f"Title: {result['title']}", file=sys.stderr)
    if result.get("channel"):
        print(f"Channel: {result['channel']}", file=sys.stderr)

    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.output == "txt":
        print(result["text"])
    else:  # srt
        print(to_srt(result["chunks"]))


if __name__ == "__main__":
    main()
