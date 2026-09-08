from vertexai.language_models import TextEmbeddingModel

model = None
BATCH_SIZE = 50
MAX_BATCH_CHARS = 36_000
MAX_TEXT_CHARS = 6_000

def _batched(texts: list[str]):
    """Yields batches bounded by both instance count and total characters."""
    batch, batch_chars = [], 0

    for t in texts:
        if batch and (len(batch)> BATCH_SIZE or batch_chars + len(t)> MAX_BATCH_CHARS):
            yield batch
            batch, batch_chars = [], 0

        batch.append(t)
        batch_chars +=len(t)

    if batch:
        yield batch


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

    model = get_embedding_model()
    embeddings= model.get_embeddings([query])
    return embeddings[0].values

def embed_texts(texts : list[str]):
    """
    Embeds a list of text strings in batches
    """
    model = get_embedding_model()
    all_embeddings = []

    """for i in range(0,len(texts), BATCH_SIZE):
    
        batch = texts[i: i+BATCH_SIZE] """
    for batch in _batched([t[:MAX_TEXT_CHARS] for t in texts]):
        embeddings = model.get_embeddings(batch)
        all_embeddings.extend( [e.values for e in embeddings])

    return all_embeddings

