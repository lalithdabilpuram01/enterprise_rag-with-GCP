import logfire
from langchain_groq import ChatGroq
from app.agents.state import Agenstate
from app.config import settings


llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model = settings.GROQ_MODEL,
    temperature=0.1
)

def generate_node(state: Agenstate):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.
        
        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else :
        logfire.info("Generating technical RAG Response.")
        max_context_chars = 25000
        full_context = ""
        for doc in state["documents"]:
            if len(full_context)+len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else : 
                logfire.warning("Context truncated to fit Groq TPM limits. ")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        
        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:

        "{user_msg}"

        """

    with logfire.span("LLM synthesis"):
        try:
            response = llm.invoke(prompt)
            logfire.info("Response synthesized successfully.")
            return {
                "final_answer": response.content,
                "status": "Response generated.",
                "messages": [{"role": "assistant", "content": response.content}]
            }
        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e