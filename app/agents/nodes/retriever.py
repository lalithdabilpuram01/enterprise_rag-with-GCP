import logfire
from app.agents.state import Agenstate
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: Agenstate):
    """
    Skips retrieval if the planner detected a conversational/memory based query.
    """

    query = state['current_query']

    if query == "CONVERSATIONAL":

        logfire.info("Skipping retrieval - query is COVERSATIONAL")

        return {
            "documents" : [],
            "status": "Using conversation history. No retrieval needed.",
            "plan": state["plan"] + ["Retrieval Skipped"]


        }
    # Standard Retrieval Logic
    with logfire.span("Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from vector database")

        doc_contents = [doc['content'] for doc in raw_results]

        with logfire.span("Semantic Reranking"):

            rerank_contents = rerank_documents(query, doc_contents, top_n= 5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [f"CONTENT: {docs}" for docs in rerank_contents]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["context Retrived"]
    }
