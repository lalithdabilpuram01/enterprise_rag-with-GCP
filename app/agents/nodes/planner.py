from langchain_groq import ChatGroq
from app.agents.state import Agentstate
from app.config import settings
import logfire

#print(str(settings.GROQ_MODEL))
#print(str(settings.GROQ_API_KEY))
# Initialize the groq model
llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    
    temperature=0.2,
    max_tokens=512,
)

def planner_node( state: Agentstate):
    """
    The planner determines if a search is needed based on the ENTIRE conversation.
    """

    history = ""

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query. 
    """

    with logfire.span("Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return{
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory).....",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for : {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }


if __name__ == "__main__":
    print("testing..")
    def test_function(user_message_1: str):
        """
            The planner determines if a search is needed based on the ENTIRE conversation.
            """
        history_1 = ""

        prompt_1 = f"""
            You are an intelligent Assistant Planner.
            Analyze the conversation history and the latest user message.
        
            CONVERSATION HISTORY:
            {history_1}
        
            LATEST MESSAGE:
            "{user_message_1}"
        
            Task:
            
            1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
            2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, then output a refined search query .
            
            Output ONLY 'CONVERSATIONAL' or the search query. 
            """
        dec = llm.invoke(prompt_1).content.strip()
        print(dec)
        if dec == "CONVERSATIONAL":
                return{
                    "current_query": "CONVERSATIONAL",
                    "status": "Handling conversationally (using memory).....",
                    "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
                }
        
        return {
                "current_query": dec,
                "status": f"Technical research needed. Searching for : {dec}",
                "plan": ["Intent: Technical", f"Search Term: {dec}"]
            }

    print(test_function("how to autoscale pods in kubernetes"))  



        
    
    