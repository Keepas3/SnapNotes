from app.llm.base import LLM
from app.llm.client import ResearchResult
from app.llm.prompts import enrich_task
from app.llm.schemas import UnderstandResult


def enrich(client: LLM, understanding: UnderstandResult) -> ResearchResult:
    """Web research from the understanding summary only (the notes themselves are not re-sent).

    Claude always researches. Gemini's Google Search is used only when the fact check found statements that look wrong,
    so an ordinary document makes no search call and sends nothing to Google beyond the notes themselves."""
    suspect = list(getattr(understanding, "suspect_claims", ()) or ())
    if client.cfg.provider == "gemini" and not suspect:
        return ResearchResult(text="", evidence=[], searches_used=0)
    task = enrich_task(understanding.main_topic, understanding.gaps, understanding.search_queries, suspect)
    return client.research("enrich", client.cfg.enrich, content=[{"type": "text", "text": task}])
