from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import Agenstate
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# 1. Initilize the State Graph

workflow = StateGraph(Agenstate)

# 2. Define the Nodes

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

# 3. define the edges & Routing Logic

def route_planner(state: Agenstate):
    """
    Routes the workflow based on the planner's decision.
    """

    if state["current_query"]== "CONVERSATIONAL" :
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")

# conditional Edge: Planner -> router -> (Retriever OR Responder)

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {"retriever": "retriever",
     "responder": "responder"}
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

# MEMORY UPGRADE
# MemorySaver allows the agent to remember conversations based on "threads".

checkpointer = MemorySaver()

# 4. compile the Graph with Memory
rag_agent = workflow.compile(checkpointer= checkpointer)
