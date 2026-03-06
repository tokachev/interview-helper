import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_deepgram_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise ValueError("DEEPGRAM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
# 150ms chunks: small enough for low latency, large enough to amortize overhead
CHUNK_FRAMES = int(SAMPLE_RATE * 0.15)  # 2400 frames

# Deepgram connection settings
DEEPGRAM_MODEL = "nova-2"
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "en")
DEEPGRAM_OPTIONS = {
    "model": DEEPGRAM_MODEL,
    "language": DEEPGRAM_LANGUAGE,
    "punctuate": True,
    "interim_results": True,
    "endpointing": 300,
}

# Device name fragment to look for when auto-detecting BlackHole
BLACKHOLE_NAME_FRAGMENT = "BlackHole"

# Output directory for transcripts
TRANSCRIPT_DIR = _PROJECT_ROOT / "transcripts"


# --- Claude API (OAuth) ---
def get_anthropic_token() -> str:
    token = os.environ.get("ANTHROPIC_TOKEN", "")
    if not token:
        raise ValueError("ANTHROPIC_TOKEN is not set in .env")
    return token


CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MAX_TOKENS = 4096
