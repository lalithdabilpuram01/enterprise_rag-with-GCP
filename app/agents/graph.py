from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import Agentstate
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.grader import grader_node, route_grader


# 1. Initilize the State Graph

workflow = StateGraph(Agentstate)

# 2. Define the Nodes

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("grader", grader_node)

# 3. define the edges & Routing Logic

def route_planner(state: Agentstate):
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
workflow.add_edge("responder", "grader")

# conditional Edge: Grader -> (Retriever for another attempt OR END)

workflow.add_conditional_edges(
    "grader",
    route_grader,
    {"retriever": "retriever",
     "end": END}
)

# MEMORY UPGRADE
# MemorySaver allows the agent to remember conversations based on "threads".

checkpointer = MemorySaver()

# 4. compile the Graph with Memory
rag_agent = workflow.compile(checkpointer= checkpointer)
