import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # --- GCP CONFIG --
    PROJECT_ID = os.getenv("PROJECT_ID", "enterprise-rag-1")
    LOCATION = os.getenv("LOCATION", "us-central1")
    GCP_DOC_AI_LOCATION = os.getenv("GCP_DOC_AI_LOCATION", "us")
    GCP_DOC_AI_PROCESSOR_ID = os.getenv("GCP_DOC_AI_PROCESSOR_ID")
    RAW_BUCKET = os.getenv("GCP_RAW_BUCKET", "enterprise-rag-1-raw")
    PROCESSED_BUCKET = os.getenv("GCP_PROCESSED_BUCKET", "enterprise-rag-1-processed")
    VPN_CONNECTOR = os.getenv("VPC_CONNECTOR", "rag-vpc")

    # --- VECTOR DB ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- EMBEDDINGS (Vertex AI online-prediction quota controls) ---
    # Vertex AI caps online prediction requests per minute per model/region.
    # Ingesting a large PDF issues one request per chunk-batch back to back, so
    # these bound the client-side request rate and the retry behaviour on 429s.
    EMBED_REQUESTS_PER_MINUTE = int(os.getenv("EMBED_REQUESTS_PER_MINUTE", "100"))
    EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))
    EMBED_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_BASE_DELAY", "2.0"))
    EMBED_RETRY_MAX_DELAY = float(os.getenv("EMBED_RETRY_MAX_DELAY", "60.0"))

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip().strip(",")
    # Groq free tier caps each model at 8000 tokens/minute (input + output), so every
    # prompt must stay well under that or Groq rejects it with 429 "Request too large".
    GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))

    # --- RERANKER ---
    # "vertex" = Vertex AI Ranking API, "flashrank" = local ONNX cross-encoder.
    # FlashRank is always the fallback if the Vertex call fails.
    RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "vertex").strip().lower()
    VERTEX_RANKER_MODEL = os.getenv("VERTEX_RANKER_MODEL", "semantic-ranker-default@latest")
    FLASHRANK_CACHE_DIR = os.getenv("FLASHRANK_CACHE_DIR", "/tmp/flashrank")

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprise-rag-1")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

# Apply Langchain environment variables for automatic tracing.
# os.environ rejects None, so unset optional vars are skipped rather than crashing on import.
_LANGCHAIN_ENV = {
    "LANGCHAIN_TRACING_V2": Settings.LANGSMITH_TRACING,
    "LANGCHAIN_API_KEY": Settings.LANGSMITH_API_KEY,
    "LANGCHAIN_PROJECT": Settings.LANGSMITH_PROJECT,
    "LANGCHAIN_ENDPOINT": Settings.LANGSMITH_ENDPOINT,
}

for _key, _value in _LANGCHAIN_ENV.items():
    if _value is not None:
        os.environ[_key] = _value




settings = Settings()
