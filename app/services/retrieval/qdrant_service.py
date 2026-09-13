import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import get_embedding_model, embed_query

# Initiliaze Qdrant Client

client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key= settings.QDRANT_API_KEY
)


def search_enterprise_knowledge(query: str, limit : int = 5):
    """
    Performs a high- precision search in the enterprise knwoledge base.
    Uses the modern query_points interface."""

    try: 
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant

        response = client.query_points(
            collection_name= settings.QDRANT_COLLECTION,
            query= query_vector,
            limit=limit,
            with_payload=True # Returns a JSON object
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text",""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
                
            })

        return results

    except Exception as e:
        logfire.error(f"Qdrant Search failed: {e}")

        return []
