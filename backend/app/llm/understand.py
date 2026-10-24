from app.core.config import fact_check_enabled
from app.ingest.loader import SourceDoc
from app.llm.base import LLM
from app.llm.prompts import understand_task
from app.llm.schemas import UnderstandResult, UnderstandResultChecked


def understand(client: LLM, source: SourceDoc) -> UnderstandResult:
    checked = fact_check_enabled(client.cfg)  # also asks for statements in the notes that look factually wrong
    return client.parse(
        "understand",
        client.cfg.understand,  # the client's own settings, so the model that is priced is the model that is used
        content=source.content(understand_task(fact_check=checked)),
        output_model=UnderstandResultChecked if checked else UnderstandResult,
    )
