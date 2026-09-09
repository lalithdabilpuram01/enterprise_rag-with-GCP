from typing import TypedDict, List, Annotated
import operator

class Agenstate(TypedDict):
    # Using the Annotated with operator.
    # are appended to the history rather than rep

    messages : Annotated[List[dict], operator.add]
    current_query : str
    documents : List[str]
    plan : List[str]
    status: str
    final_answer : str