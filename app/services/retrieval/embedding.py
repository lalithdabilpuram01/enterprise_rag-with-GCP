import random
import threading
import time

import logfire
from google.api_core import exceptions as google_exceptions
from vertexai.language_models import TextEmbeddingModel

from app.config import settings

model = None
BATCH_SIZE = 50
MAX_BATCH_CHARS = 36_000
MAX_TEXT_CHARS = 6_000

# Vertex AI rejects online prediction requests that exceed the per-minute quota
# for the model. Spacing requests out client-side keeps a large ingestion run
# under that ceiling instead of relying on retries alone.
_MIN_REQUEST_INTERVAL = (
    60.0 / settings.EMBED_REQUESTS_PER_MINUTE
    if settings.EMBED_REQUESTS_PER_MINUTE > 0
    else 0.0
)

# Transient server-side conditions. 429 (ResourceExhausted) is the quota error.
_RETRYABLE_ERRORS = (
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
)

_throttle_lock = threading.Lock()
_last_request_at = 0.0


def _batched(texts: list[str]):
    """Yields batches bounded by both instance count and total characters."""
    batch: list[str] = []
    batch_chars = 0

    for t in texts:
        if batch and (len(batch) >= BATCH_SIZE or batch_chars + len(t) > MAX_BATCH_CHARS):
            yield batch
            batch, batch_chars = [], 0

        batch.append(t)
        batch_chars += len(t)

    if batch:
        yield batch


def _throttle():
    """Blocks until at least _MIN_REQUEST_INTERVAL has passed since the last call."""
    if _MIN_REQUEST_INTERVAL <= 0:
        return

    global _last_request_at
    with _throttle_lock:
        wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _get_embeddings(batch: list[str]):
    """
    Calls Vertex AI for one batch, throttled and retried with exponential backoff.

    A quota rejection is transient, so a batch must not fail the whole file on the
    first 429 -- it is retried with a widening delay before giving up.
    """
    active_model = get_embedding_model()

    for attempt in range(settings.EMBED_MAX_RETRIES + 1):
        _throttle()
        try:
            return active_model.get_embeddings(batch)

        except _RETRYABLE_ERRORS as e:
            if attempt == settings.EMBED_MAX_RETRIES:
                logfire.error(
                    f"Embedding batch failed after {attempt + 1} attempts: {e}"
                )
                raise

            delay = min(
                settings.EMBED_RETRY_BASE_DELAY * (2 ** attempt),
                settings.EMBED_RETRY_MAX_DELAY,
            )
            # Jitter keeps concurrent workers from retrying in lockstep.
            delay += random.uniform(0, delay * 0.25)

            logfire.warning(
                f"Vertex AI embedding request failed ({type(e).__name__}), "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{settings.EMBED_MAX_RETRIES})"
            )
            time.sleep(delay)


def get_embedding_model():
    global model
    if model is None:
        # Reverting to TextEmbeddingModel for stability
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    return model


def embed_query(query: str):
    """
    Embeds a single query string using the stable vertex AI API
    """
    embeddings = _get_embeddings([query])
    return embeddings[0].values


def embed_texts(texts: list[str]):
    """
    Embeds a list of text strings in batches
    """
    all_embeddings = []

    for batch in _batched([t[:MAX_TEXT_CHARS] for t in texts]):
        embeddings = _get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings])

    return all_embeddings
