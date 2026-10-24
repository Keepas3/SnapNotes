"""The command-line spike: it stops spending at the same between-stages limit the server uses."""
import dataclasses
import json

import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.llm.schemas import CardSet, UnderstandResult, VerifyResult
from tests.conftest import FakeSDK, citation, response, text_block
from tests.test_pipeline import UNDERSTANDING, make_card, verdict

spike = pytest.importorskip("spike")  # spike.py is not part of the Docker build context


def fake_sdk() -> FakeSDK:
    return FakeSDK(
        parse_results={
            UnderstandResult: UNDERSTANDING,
            CardSet: CardSet(cards=[make_card("What do mitochondria contain?")]),
            VerifyResult: VerifyResult(verdicts=[verdict(0)]),
        },
        create_responses=[response([text_block("- Mitochondria produce ATP.", [citation("https://a.org", "A", "produce ATP")])])],
    )


def run_spike(monkeypatch, tmp_path, cfg: Settings, sdk: FakeSDK) -> int:
    notes = tmp_path / "notes.md"
    notes.write_text("# Cells\nMitochondria contain DNA.")
    monkeypatch.setattr(spike, "build_llm", lambda _settings: LLMClient(sdk, cfg))
    monkeypatch.setattr("sys.argv", ["spike.py", str(notes), "--out", str(tmp_path / "out")])
    return spike.main()


def test_a_run_inside_the_limit_completes_and_saves_its_result(monkeypatch, tmp_path, capsys):
    assert run_spike(monkeypatch, tmp_path, Settings(), fake_sdk()) == 0
    out = capsys.readouterr().out
    assert "Cost limit $2.00" in out and "Kept 1/1 cards" in out
    saved = json.loads((tmp_path / "out" / "notes.json").read_text(encoding="utf-8"))
    assert len(saved["cards"]) == 1 and len(saved["usage"]) == 4


def test_a_run_over_the_limit_stops_before_the_next_stage_and_says_what_was_spent(monkeypatch, tmp_path, capsys):
    # price the model so absurdly that each fake call costs $1.40: the ceiling ($2.00) is passed after the second call
    expensive = dataclasses.replace(Settings(), max_job_cost_usd=2.0, extra_prices={"claude-opus-5-5": (4000.0, 20000.0)})
    sdk = fake_sdk()
    assert run_spike(monkeypatch, tmp_path, expensive, sdk) == 1
    captured = capsys.readouterr()
    assert "exceeded the $2.00 cost limit" in captured.err
    assert "--- Usage ---" in captured.out and "Estimated total" in captured.out  # the spend so far is still shown
    assert not any(kw.get("output_format") is CardSet for _, kind, kw in sdk.calls if kind == "parse")  # card generation never ran
    assert not (tmp_path / "out").exists()  # no partial result is saved


def test_the_limit_comes_from_the_same_setting_as_the_server(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SNAPNOTES_MAX_JOB_COST_USD", "0.5")
    assert Settings().max_job_cost_usd == 0.5
    run_spike(monkeypatch, tmp_path, Settings(), fake_sdk())
    assert "Cost limit $0.50" in capsys.readouterr().out
