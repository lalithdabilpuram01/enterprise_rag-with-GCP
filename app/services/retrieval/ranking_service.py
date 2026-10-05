import time
import logfire
from app.config import settings
from flashrank import Ranker, RerankRequest
from google.cloud import discoveryengine_v1 as discoveryengine

# Lazy initialization - clients are created on first use to ensure logfire is configured
_ranker = None
_vertex_client = None

def _get_flashrank() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-TinyBERT-L-2-v2) baked into the Docker image.
    """

    global _ranker
    if _ranker is None:
        logfire.info("Initializing FlashRank Model (TinyBERT) local")

        # Model is baked into the Docker image at this path; downloading at runtime
        # from Cloud Run gets rate-limited (429) by HuggingFace.
        _ranker = Ranker(cache_dir=settings.FLASHRANK_CACHE_DIR)

    return _ranker

def _get_vertex_client() -> discoveryengine.RankServiceClient:
    """
    Initializes the Vertex AI Ranking API client lazily (uses Application Default Credentials).
    """

    global _vertex_client
    if _vertex_client is None:
        logfire.info("Initializing Vertex AI Ranking client")
        _vertex_client = discoveryengine.RankServiceClient()

    return _vertex_client

def _rerank_vertex(query: str, documents: list[str], top_n: int) -> list[str]:
    client = _get_vertex_client()

    # The Ranking API is only served from the "global" location, not settings.LOCATION
    ranking_config = client.ranking_config_path(
        project=settings.PROJECT_ID,
        location="global",
        ranking_config="default_ranking_config",
    )

    request = discoveryengine.RankRequest(
        ranking_config=ranking_config,
        model=settings.VERTEX_RANKER_MODEL,
        top_n=top_n,
        query=query,
        records=[
            discoveryengine.RankingRecord(id=str(i), content=doc)
            for i, doc in enumerate(documents)
        ],
    )

    response = client.rank(request=request)

    # Records are returned sorted by highest score first; map ids back to the original docs
    top_score = response.records[0].score if response.records else 'N/A'
    logfire.info(f"[Reranker] Vertex top semantic score: {top_score}")

    return [documents[int(record.id)] for record in response.records]

def _rerank_flashrank(query: str, documents: list[str], top_n: int) -> list[str]:
    ranker = _get_flashrank()

    # FlashRank expects a list of dictionaries with 'id' and 'text'
    passages = [
        {"id": i, "text": doc}
        for i, doc in enumerate(documents)
    ]

    results = ranker.rerank(RerankRequest(query=query, passages=passages))

    # Results are returned sorted by highest semantic scores first
    top_score = results[0]['score'] if results else 'N/A'
    logfire.info(f"[Reranker] FlashRank top semantic score: {top_score}")

    return [res['text'] for res in results[:top_n]]

def rerank_documents(query: str, documents : list[str], top_n: int= 5)-> list[str]:
    """
    Refines retrieval results by rescoring documents against the query semantically.
    Uses settings.RERANKER_PROVIDER, falling back to FlashRank if Vertex fails.
    """
    if not documents:
        return []

    start_time = time.time()
    provider = settings.RERANKER_PROVIDER
    logfire.info(f"[Reranker] Sending {len(documents)} docs to {provider}...")

    if provider == "vertex":
        try:
            with logfire.span("vertex_rerank"):
                reranked_docs = _rerank_vertex(query, documents, top_n)

            logfire.info(f"[Reranker] Vertex done in {time.time() - start_time:.2f}s")
            return reranked_docs

        except Exception as e:
            logfire.error(f"[Reranker] Vertex reranking failed, falling back to FlashRank: {e}")

    try:
        with logfire.span("flashrank_rerank"):
            reranked_docs = _rerank_flashrank(query, documents, top_n)

        logfire.info(f"[Reranker] FlashRank done in {time.time() - start_time:.2f}s")
        return reranked_docs

    except Exception as e:
        logfire.error(f"[Reranker] Semantic Reranking Failed: {e}")

        return documents[:top_n]
