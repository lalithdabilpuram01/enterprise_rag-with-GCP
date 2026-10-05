import logfire
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from app.agents.state import Agentstate
from app.config import settings

# Number of times a rejected answer is sent back to the retriever before giving up
MAX_RETRIES = 3


class GradeResult(BaseModel):
    relevant: bool = Field(description="True if the answer directly and correctly addresses the user's question.")
    reason: str = Field(description="One sentence explaining the decision.")
    rewritten_query: str = Field(
        default="",
        description="If not relevant, a better search query to retrieve documentation that answers the question. Empty otherwise.",
    )


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.0,
    max_tokens=512,
)

grader_llm = llm.with_structured_output(GradeResult)


def accept_answer(state: Agentstate, status: str, plan_step: str):
    """
    Passes the answer to the output and records it in the conversation history.
    retry_count is reset because the checkpointer carries state across turns of a thread.
    """
    return {
        "status": status,
        "plan": state["plan"] + [plan_step],
        "messages": [{"role": "assistant", "content": state["final_answer"]}],
        "retry_count": 0,
    }


def grader_node(state: Agentstate):
    """
    Grades whether the responder's answer is relevant to the user's question.
    Relevant answers go to the output; irrelevant ones are sent back to the retriever
    with a rewritten search query, up to MAX_RETRIES times.
    """
    query = state["current_query"]
    answer = state["final_answer"]
    retry_count = state.get("retry_count", 0)

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    # Conversational answers come from memory, so there is nothing to re-retrieve
    if query == "CONVERSATIONAL":
        logfire.info("Skipping grading - query is CONVERSATIONAL")
        return accept_answer(state, "Response generated.", "Grading: Skipped (conversational)")

    prompt = f"""
    You are a strict answer grader for an Enterprise technical assistant.
    Decide whether the ANSWER is relevant to the USER QUESTION: it must directly address what was asked,
    not dodge it, and not say the information is unavailable.

    USER QUESTION:
    "{user_msg}"

    SEARCH QUERY USED:
    "{query}"

    ANSWER:
    {answer}

    If the answer is NOT relevant, write a rewritten_query that is different from the search query used
    and more likely to retrieve documentation that answers the question.
    """

    with logfire.span("Answer Grading", attempt=retry_count + 1):
        try:
            grade = grader_llm.invoke(prompt)
        except Exception as e:
            # A grader failure should not block an answer the user is waiting for
            logfire.error(f"Grader failed, passing answer through: {e}")
            return accept_answer(state, "Response generated (grading unavailable).", "Grading: Failed, answer passed through")

        logfire.info(f"Relevant: {grade.relevant}. Reason: {grade.reason}")

    if grade.relevant:
        return accept_answer(state, "Response verified as relevant.", "Grading: Relevant")

    if retry_count >= MAX_RETRIES:
        logfire.warning(f"Max retries ({MAX_RETRIES}) reached. Returning best available answer.")
        return accept_answer(
            state,
            f"Could not verify answer after {MAX_RETRIES} retries. Returning best available answer.",
            f"Grading: Not relevant, max retries ({MAX_RETRIES}) reached",
        )

    new_query = grade.rewritten_query.strip() or query
    logfire.info(f"Retry {retry_count + 1}/{MAX_RETRIES}. Re-searching for: {new_query}")

    return {
        "current_query": new_query,
        "retry_count": retry_count + 1,
        "status": f"Answer not relevant. Retrying search ({retry_count + 1}/{MAX_RETRIES}) for : {new_query}",
        "plan": state["plan"] + [f"Grading: Not relevant ({grade.reason})", f"Retry Search Term: {new_query}"],
    }


def route_grader(state: Agentstate):
    """
    Routes to the retriever if the grader asked for another search, otherwise to the output.
    """
    # grader_node only leaves retry_count above 0 when it rejected the answer
    if state.get("retry_count", 0) > 0:
        return "retriever"
    return "end"
