"""Canned Claude responses for developing the mobile app without an API key or any spend.

Enabled with SNAPNOTES_DEMO=1. It plugs in where the Anthropic SDK normally sits (see LLMClient), so the real
worker, database, approval step and API all run unchanged. Output does not depend on the uploaded notes except
for the deck title, which is taken from the first line of text notes.
"""
from __future__ import annotations

import re
import time
from types import SimpleNamespace
from typing import Any

from app.llm.schemas import (
    Card,
    CardSet,
    CardVerdict,
    CardVerdictChecked,
    UnderstandResult,
    UnderstandResultChecked,
    VerifyResult,
    VerifyResultChecked,
)


def _usage(inp: int, out: int, searches: int = 0):
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out, cache_read_input_tokens=0, cache_creation_input_tokens=0,
        server_tool_use=SimpleNamespace(web_search_requests=searches),
    )


def _topic_from(messages: list[dict]) -> str:
    for block in messages[0]["content"]:
        text = block.get("text", "") if isinstance(block, dict) else ""
        if text.startswith("<notes"):
            for line in text.splitlines()[1:]:
                line = re.sub(r"^[#>\-*\s]+", "", line).strip()
                if line:
                    return line[:60]
    return "Cell biology basics"


def _last_text(messages: list[dict]) -> str:
    """The stage instructions: the last text block of the request."""
    blocks = messages[0]["content"]
    return blocks[-1].get("text", "") if blocks and isinstance(blocks[-1], dict) else ""


def _card(type_: str, front: str, back: str, origin: str, ref: str, snippet: str, tags: list[str], difficulty: str = "medium") -> Card:
    return Card(type=type_, front=front, back=back, origin=origin, source_ref=ref, source_snippet=snippet, tags=tags, difficulty=difficulty)  # type: ignore[arg-type]


_WIKI = "https://en.wikipedia.org/wiki/"

DEMO_CARDS = [
    _card("basic", "What is the main function of mitochondria?", "They produce most of the cell's ATP through cellular respiration.", "notes", "p.1", "Mitochondria are the powerhouse of the cell", ["mitochondria"], "easy"),
    _card("cloze", "The Krebs cycle takes place in the {{c1::mitochondrial matrix}}.", "It produces NADH and FADH2, which feed the electron transport chain.", "notes", "p.1", "The Krebs cycle takes place in the mitochondrial matrix", ["metabolism"]),
    _card("definition", "Osmosis", "Movement of water across a semipermeable membrane from low to high solute concentration.", "notes", "p.2", "movement of water across a semipermeable membrane", ["transport"], "easy"),
    _card("basic", "From which parent is mitochondrial DNA usually inherited?", "Almost always the mother.", "web", _WIKI + "Mitochondrial_DNA", "mtDNA is inherited almost exclusively through the maternal line", ["mitochondria", "genetics"]),
    _card("basic", "What are ribosomes made of?", "Ribosomal RNA (rRNA) and proteins. The notes say 'protein only', which is incorrect.", "web", _WIKI + "Ribosome", "ribosomes consist of rRNA and proteins", ["ribosome"], "hard"),
    _card("cloze", "Water moves by osmosis from a region of {{c1::low}} to {{c2::high}} solute concentration.", "Water follows the solute gradient across the membrane.", "notes", "p.2", "low solute concentration to high solute concentration", ["transport"]),
    _card("basic", "What is the primary function of mitochondria?", "They generate ATP.", "notes", "p.1", "powerhouse of the cell", ["mitochondria"], "easy"),  # near-duplicate of #0
    _card("basic", "How many mitochondria does a typical liver cell contain?", "Around 2000.", "notes", "p.3", "", ["mitochondria"], "hard"),  # not supported by anything
]

# A correction made from the AI's own knowledge, added only when the card prompt lists doubtful statements (SNAPNOTES_FACT_CHECK=1).
DEMO_AI_CARD = _card(
    "basic", "In which kinds of cells are mitochondria found?", "Nearly all plant and animal cells. (The notes said plant cells only.)",
    "ai", "Corrects: Mitochondria are found only in plant cells", "", ["mitochondria"], "easy",
)

# What the fact check would say about a card (card index -> concern), used only when SNAPNOTES_FACT_CHECK=1 in demo mode.
DEMO_CONCERNS = {3: "Rare cases of paternal inheritance of mitochondrial DNA have been reported, so 'only the mother' is a simplification."}

# (claim, url, title, passage the claim rests on)
DEMO_EVIDENCE = [
    ("Mitochondrial DNA is inherited almost exclusively from the mother.", _WIKI + "Mitochondrial_DNA", "Mitochondrial DNA - Wikipedia", "In most multicellular organisms, mtDNA is inherited from the mother (maternally inherited)."),
    ("Ribosomes consist of ribosomal RNA and proteins, not protein alone.", _WIKI + "Ribosome", "Ribosome - Wikipedia", "Ribosomes are made from complexes of RNAs and proteins."),
    ("In eukaryotes the citric acid (Krebs) cycle runs in the mitochondrial matrix.", _WIKI + "Citric_acid_cycle", "Citric acid cycle - Wikipedia", "In eukaryotic cells, the citric acid cycle occurs in the matrix of the mitochondrion."),
]


class _Namespace:
    def __init__(self, delay: float):
        self.delay = delay

    def _pause(self) -> None:
        if self.delay:
            time.sleep(self.delay)

    def parse(self, **kw: Any):
        self._pause()
        model = kw["output_format"]
        if model in (UnderstandResult, UnderstandResultChecked):
            fields: dict[str, Any] = dict(
                main_topic=_topic_from(kw["messages"]),
                subject_area="biology",
                subtopics=["Mitochondria", "Metabolism", "Membrane transport", "Ribosomes"],
                key_terms=["ATP", "Krebs cycle", "osmosis", "ribosome"],
                gaps=["Notes claim ribosomes are made of protein only", "Mitochondrial inheritance is mentioned without detail"],
                search_queries=["ribosome composition rRNA protein", "mitochondrial DNA inheritance", "Krebs cycle location"],
            )
            if model is UnderstandResultChecked:  # the fact check is on (SNAPNOTES_FACT_CHECK=1)
                fields["suspect_claims"] = [
                    "Ribosomes are made of protein only — ribosomes contain ribosomal RNA as well as protein.",
                    "Mitochondria are found only in plant cells — they are found in nearly all plant and animal cells.",
                ]
            out: Any = model(**fields)
            usage = _usage(1800, 350)
        elif model is CardSet:
            doubtful = "<doubtful_statements>" in _last_text(kw["messages"])
            out, usage = CardSet(cards=DEMO_CARDS + [DEMO_AI_CARD] if doubtful else DEMO_CARDS), _usage(2200, 1400)
        elif model in (VerifyResult, VerifyResultChecked):
            checked = model is VerifyResultChecked
            verdict_cls = CardVerdictChecked if checked else CardVerdict
            count = len(DEMO_CARDS) + (1 if '"origin": "ai"' in _last_text(kw["messages"]) else 0)  # the AI correction, when there is one
            out = model(verdicts=[
                verdict_cls(
                    index=i, atomic=True, grounded=(i != 7),
                    issue="Not supported by the notes or the evidence" if i == 7 else "",
                    duplicate_of=0 if i == 6 else -1,  # card 6 repeats card 0
                    **({"factual_concern": DEMO_CONCERNS.get(i, "")} if checked else {}),
                )
                for i in range(count)
            ])
            usage = _usage(2600, 300)
        else:  # pragma: no cover - guards against a new stage being added without demo data
            raise NotImplementedError(f"No demo data for {model}")
        return SimpleNamespace(parsed_output=out, usage=usage, stop_reason="end_turn", content=[], stop_details=None)

    def create(self, **kw: Any):
        """The web research call."""
        self._pause()
        blocks: list[Any] = [SimpleNamespace(type="web_search_tool_result", content=[SimpleNamespace(type="web_search_result")])]
        for claim, url, title, passage in DEMO_EVIDENCE:
            cit = SimpleNamespace(type="web_search_result_location", url=url, title=title, cited_text=passage)
            blocks.append(SimpleNamespace(type="text", text=f"- {claim}\n", citations=[cit]))
        return SimpleNamespace(content=blocks, stop_reason="end_turn", usage=_usage(900, 450, searches=3), stop_details=None)


class DemoSDK:
    """Quacks like `anthropic.Anthropic()` for the handful of calls LLMClient makes."""

    def __init__(self, delay: float = 1.2):
        self.messages = _Namespace(delay)
        self.beta = SimpleNamespace(messages=_Namespace(delay))
