"""
Human-in-the-loop eval: wire user feedback to LangSmith traces.

Every thumbs up/down from the UI is recorded against the exact RAG run
that produced the response, giving you labelled production data for free.
Thumbs-down runs can be exported and promoted into the golden eval set.
"""
import os
from typing import Optional

from langsmith import Client

_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LANGCHAIN_API_KEY not set — HITL eval requires LangSmith."
            )
        _client = Client(api_key=api_key)
    return _client


def record_user_feedback(
    run_id: str,
    thumbs_up: bool,
    comment: str = "",
) -> None:
    """
    Record UI feedback against the LangSmith run that produced the answer.
    Call this from the POST /api/feedback endpoint.

    run_id: the LangSmith run ID returned with the search response.
    thumbs_up: True = positive, False = negative.
    comment: optional free-text from the user.
    """
    _get_client().create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=1 if thumbs_up else 0,
        comment=comment,
        feedback_source_type="app",
    )


def export_negative_feedback(limit: int = 100) -> list:
    """
    Pull thumbs-down runs from LangSmith and return them as labelled eval cases.

    Run this periodically (e.g. weekly) to grow your regression suite from
    real failures rather than manually crafting examples.

    Returns a list of dicts ready to feed into model_grader.run_eval_suite().
    """
    client = _get_client()
    feedback_items = client.list_feedback(
        feedback_key=["user_satisfaction"],
        limit=limit,
    )

    eval_cases = []
    for fb in feedback_items:
        if fb.score != 0:
            continue  # keep thumbs-down only

        try:
            run = client.read_run(fb.run_id)
            eval_cases.append({
                "run_id": str(fb.run_id),
                "question": run.inputs.get("query", ""),
                "context": run.inputs.get("context", ""),
                "answer": (run.outputs or {}).get("response", ""),
                "user_comment": fb.comment or "",
            })
        except Exception:
            continue  # run may have been pruned; skip silently

    return eval_cases


def promote_to_golden_set(
    eval_cases: list,
    dataset_name: str = "rag-golden-eval-set",
) -> str:
    """
    Add reviewed eval cases to a named LangSmith dataset for ongoing regression testing.

    Typical workflow:
      1. cases = export_negative_feedback()
      2. Manually review — remove spam, add expected_answer where known
      3. promote_to_golden_set(cases)
      4. Run model_grader.run_eval_suite() against the dataset in CI

    Returns the dataset ID.
    """
    client = _get_client()

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden eval set built from production thumbs-down feedback",
        )

    client.create_examples(
        inputs=[{"question": c["question"], "context": c["context"]} for c in eval_cases],
        outputs=[{"expected_answer": c.get("expected_answer", "")} for c in eval_cases],
        dataset_id=dataset.id,
    )

    return str(dataset.id)
