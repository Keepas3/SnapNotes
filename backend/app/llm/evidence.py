from app.llm.schemas import EvidenceItem


def format_evidence(items: list[EvidenceItem]) -> str:
    """Approved evidence as a delimited block for the generate/verify prompts."""
    approved = [e for e in items if e.approved]
    if not approved:
        return "(no web evidence)"
    lines = []
    for i, e in enumerate(approved, 1):
        lines.append(f"[E{i}] {e.claim}\n  source: {e.title} - {e.url}\n  passage: {e.snippet[:300]}")
    return "\n".join(lines).replace("</evidence>", "<\\/evidence>")
