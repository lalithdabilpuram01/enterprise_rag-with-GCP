import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Reranker is loaded on first use to ensure logfire
_ranker = None

def _get_reranker() -> Ranker:
    """
    Initilizes the FlashRank engine lazily.
    FlashRank uses a local ONNX model(ms-marco-MiniLM-L-6-v2) for ultra
    """

    global _ranker
    if _ranker is None:
        logfire.info("Initilizing FlashRank Model (TinyBERT) local")
        try: 

            _ranker = Ranker(cache_dir="/tmp/flashrank")

        except Exception:
            _ranker = Ranker()

    return _ranker

def rerank_documents(query: str, documents : list[str], top_n: int= 5)-> list[str]:
    """
    Refines retrieval results by rescoring documents against the query Semantically
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"[Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_reranker()

        # FlashRank excepts a list of dictionaires with 'id' and 'text'

        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages= passages)
        results = ranker.rerank(request)

        reranked_docs = []
        # Results are returned sorted by higest semantic scores first
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'

        logfire.info(f"[Reranker] Done in {duration:.2f}s. Top Semantic score: {top_score}")
        

        return reranked_docs

    except Exception as e:
        logfire.error(f"[Reranker] Semantic Reranking Failed: {e}")

        return documents[:top_n]


