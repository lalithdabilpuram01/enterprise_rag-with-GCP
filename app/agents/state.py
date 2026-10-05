from typing import TypedDict, List, Annotated
import operator

class Agentstate(TypedDict):
    # Using the Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced

    messages : Annotated[list[dict], operator.add]
    current_query : str
    documents : list[str]
    plan : list[str]
    status: str
    final_answer : str
    retry_count : int