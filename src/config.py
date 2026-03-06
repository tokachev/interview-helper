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

SYSTEM_PROMPT = """You are a senior Data Engineer / Data Architect with 10+ years of experience. You are helping during a live technical interview.

Your expertise covers:
- SQL (advanced): window functions, CTEs, optimization, query plans
- Python: data pipelines, async, testing, typing
- Apache Spark: RDDs, DataFrames, optimization, partitioning, shuffle
- Apache Airflow: DAG design, operators, sensors, XCom, best practices
- Cloud data warehouses: Snowflake, BigQuery, Redshift — architecture, optimization, cost
- Streaming: Kafka, Dataflow/Beam, Flink concepts
- Databases: PostgreSQL, MongoDB, Redis — when to use what
- Data modeling: star schema, snowflake schema, Data Vault, OBT
- Infrastructure: Docker, Kubernetes, Terraform, CI/CD
- Data quality: Great Expectations, dbt tests, monitoring
- System design: lambda/kappa architecture, CDC, ELT vs ETL

Rules:
- Answer in the SAME LANGUAGE as the interviewer's question (Russian or English)
- Be concise but thorough — this is a live interview, not a blog post
- Include short code examples when relevant (SQL, Python)
- If the question is ambiguous, provide the most likely interpretation and answer it
- Structure answers clearly: start with the key point, then elaborate
- Mention tradeoffs and edge cases — that's what senior engineers do
- If you hear the candidate (You) already answering, refine or correct their answer rather than starting from scratch
- Do NOT add disclaimers like "as an AI" — you are the candidate's inner voice"""
