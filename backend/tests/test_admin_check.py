"""`python -m app.admin check`: tells a beginner whether their environment variables are usable, without echoing a secret."""
import os
from types import SimpleNamespace

import anthropic
import pytest
from google.genai import errors

from app.admin import BAD, OK, WARN, main as admin, preflight
from app.core.config import Settings

TOKEN = "T" * 43
KEY = "sk-ant-" + "k" * 60


@pytest.fixture
def env(monkeypatch):
    """A clean slate: nothing from the developer's own shell leaks into these tests."""
    for name in [n for n in os.environ if n.startswith(("SNAPNOTES_", "ANTHROPIC_", "GEMINI_", "GOOGLE_"))]:
        monkeypatch.delenv(name)

    def setup(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return setup


def run(capsys, *args) -> tuple[int, str]:
    code = admin(["check", *args])
    return code, capsys.readouterr().out


def test_a_good_setup_passes_and_reveals_no_secret(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY, SNAPNOTES_MAX_JOB_COST_USD="0.5")
    code, out = run(capsys)
    assert code == 0 and "Looks good" in out
    assert "SNAPNOTES_TOKEN is set (43 characters)" in out
    assert "starts with sk-ant-, 67 characters" in out
    assert "about $0.50" in out
    assert TOKEN not in out and KEY not in out and "kkkkk" not in out and "TTTTT" not in out
    assert "check --online" in out  # points at the optional extra test


def test_no_token_is_a_problem(env, capsys):
    env(ANTHROPIC_API_KEY=KEY)
    code, out = run(capsys)
    assert code == 1 and f"{BAD} SNAPNOTES_TOKEN is not set" in out and "fix them in this window" in out


def test_a_short_token_is_a_problem_but_demo_mode_allows_it(env, capsys):
    env(SNAPNOTES_TOKEN="dev-token", ANTHROPIC_API_KEY=KEY)
    code, out = run(capsys)
    assert code == 1 and "only 9 characters" in out

    env(SNAPNOTES_DEMO="1")
    code, out = run(capsys)
    assert code == 0 and "Demo mode is on" in out and "short, allowed" in out


@pytest.mark.parametrize(
    "token", ["<" + "a" * 40 + ">", '"' + "a" * 40 + '"', "aaaa bbbb " + "c" * 30, "<the string it printed>"]  # the last: the doc's placeholder, copied as is
)
def test_placeholder_brackets_quotes_and_spaces_left_in_the_token_are_caught(env, capsys, token):
    env(SNAPNOTES_TOKEN=token, ANTHROPIC_API_KEY=KEY)
    code, out = run(capsys)
    assert code == 1 and "placeholder" in out and "SNAPNOTES_TOKEN" in out
    assert token not in out


def test_the_same_mistakes_are_caught_in_the_api_key_and_an_odd_prefix_only_warns(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY="<" + KEY + ">")
    code, out = run(capsys)
    assert code == 1 and f"{BAD} ANTHROPIC_API_KEY" in out

    env(ANTHROPIC_API_KEY="not-an-anthropic-key-0123456789")
    code, out = run(capsys)
    assert code == 0 and f"{WARN} ANTHROPIC_API_KEY is set" in out and "does not start with sk-ant-" in out
    assert "not-an-anthropic-key" not in out


NO_CREDENTIALS = (BAD, "No Claude credentials found. Set ANTHROPIC_API_KEY in this same window before starting the server.")


def client_with(**found):
    return SimpleNamespace(**{"api_key": None, "auth_token": None, "credentials": None, "custom_auth": None, **found})


def test_missing_credentials_are_reported_unless_another_login_method_works(env):
    cfg = Settings()

    def refuses():
        raise TypeError("Could not resolve authentication method")

    assert NO_CREDENTIALS in preflight(cfg, client_factory=refuses)
    assert NO_CREDENTIALS in preflight(cfg, client_factory=lambda: client_with())
    for found in ({"auth_token": "t"}, {"credentials": object()}):  # e.g. a login made with `ant auth login`
        lines = preflight(cfg, client_factory=lambda found=found: client_with(**found))
        assert any(level == OK and "credentials were found" in msg for level, msg in lines)


def test_the_real_sdk_builds_a_client_with_no_credentials_so_the_check_must_look_inside_it(env):
    """Regression: the first version asked 'did constructing the client raise?', which is always no."""
    client = anthropic.Anthropic()
    if client.credentials or client.custom_auth:
        pytest.skip("this machine has a Claude login, so credentials really are found")
    assert NO_CREDENTIALS in preflight(Settings())


def test_demo_mode_needs_no_key(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_DEMO="1")
    code, out = run(capsys)
    assert code == 0 and "no AI is called" in out and "--online" not in out


def test_a_setting_that_cannot_be_read_is_explained_instead_of_a_traceback(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY, SNAPNOTES_MAX_UPLOAD_MB="thirty")
    code, out = run(capsys)
    assert code == 1 and f"{BAD} A setting could not be read" in out


def test_an_unpriced_model_is_a_problem_until_a_price_is_given(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY, SNAPNOTES_GENERATE_MODEL="claude-mystery-1")
    code, out = run(capsys)
    assert code == 1 and "claude-mystery-1" in out and "SNAPNOTES_EXTRA_PRICES" in out

    env(SNAPNOTES_EXTRA_PRICES='{"claude-mystery-1": [1.0, 5.0]}')
    code, out = run(capsys)
    assert code == 0 and "all have a known price" in out


def test_a_zero_spend_limit_is_a_problem(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY, SNAPNOTES_MAX_JOB_COST_USD="0")
    code, out = run(capsys)
    assert code == 1 and "stopped straight away" in out


def test_checking_changes_nothing_on_disk(env, capsys, tmp_path):
    data = tmp_path / "not-created-yet"
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY, SNAPNOTES_DATA_DIR=str(data))
    code, out = run(capsys)
    assert code == 0 and "will be created on first start" in out
    assert not data.exists()  # no folder, no database


# --- --online: asking Claude whether the key works ------------------------------------------------------------


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.models = self
        self.options = {}

    def with_options(self, **options):
        self.options = options
        return self

    def list(self, **kwargs):
        if self.error:
            raise self.error
        return []


def api_error(cls, status=500, message="boom"):
    exc = cls.__new__(cls)  # the real constructors need an HTTP response object; only the class and two fields matter here
    Exception.__init__(exc, message)
    exc.status_code, exc.message = status, message
    return exc


def online_line(error):
    client = FakeClient(error)
    lines = preflight(Settings(), online=True, client_factory=lambda: client)
    return lines[-1], client


def test_online_accepts_a_working_key_quickly_and_without_retries(env):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY)
    (level, text), client = online_line(None)
    assert (level, text) == (OK, "Claude accepted the credentials.")
    assert client.options == {"max_retries": 0, "timeout": 15.0}  # a wrong network must not hang the command


@pytest.mark.parametrize(
    "error, level, fragment",
    [
        (api_error(anthropic.AuthenticationError, 401), BAD, "rejected the credentials"),
        (api_error(anthropic.PermissionDeniedError, 403), BAD, "rejected the credentials"),
        (api_error(anthropic.APIConnectionError, 0, "no network"), WARN, "Could not reach Claude"),
        (api_error(anthropic.InternalServerError, 529, "overloaded"), BAD, "(529)"),
    ],
)
def test_online_explains_each_kind_of_failure(env, error, level, fragment):
    env(SNAPNOTES_TOKEN=TOKEN, ANTHROPIC_API_KEY=KEY)
    (got_level, text), _ = online_line(error)
    assert got_level == level and fragment in text


# --- the free Gemini provider -----------------------------------------------------------------------------

GEMINI_KEY = "AIza" + "g" * 35


def test_a_gemini_setup_passes_with_its_two_warnings_and_reveals_no_secret(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gemini", GEMINI_API_KEY=GEMINI_KEY)  # no Claude key needed
    code, out = run(capsys)
    assert code == 0 and "GEMINI_API_KEY is set (39 characters)" in out
    assert "AI provider: gemini, model gemini-3.8-flash" in out and "no per-document spend limit applies" in out
    assert f"{WARN} Notes sent to the free Gemini tier may be used by Google" in out
    assert f"{WARN} Web research with Gemini (Google Search) runs only when a statement in your notes looks wrong" in out
    assert f"{OK} AI corrections: on." in out
    assert GEMINI_KEY not in out and "gggg" not in out
    assert "ANTHROPIC" not in out and "spent about" not in out  # nothing about Claude's key or dollars
    assert "To also test that Gemini accepts your key" in out


def test_the_other_google_variable_name_counts_too(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gemini", GOOGLE_API_KEY=GEMINI_KEY)
    assert run(capsys)[0] == 0


def test_a_missing_or_mispasted_gemini_key_is_a_problem(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gemini")
    code, out = run(capsys)
    assert code == 1 and f"{BAD} GEMINI_API_KEY is not set" in out and "Google AI Studio" in out

    env(GEMINI_API_KEY="<your gemini key>")
    code, out = run(capsys)
    assert code == 1 and f"{BAD} GEMINI_API_KEY:" in out and "placeholder" in out and "your gemini key" not in out


def test_an_unknown_provider_is_named(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gpt", ANTHROPIC_API_KEY=KEY)
    code, out = run(capsys)
    assert code == 1 and f"{BAD} SNAPNOTES_PROVIDER is 'gpt'; it must be one of: anthropic, gemini." in out


def test_demo_mode_wins_over_the_provider(env, capsys):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gemini", SNAPNOTES_DEMO="1")
    code, out = run(capsys)
    assert code == 0 and "no AI is called" in out and "GEMINI_API_KEY" not in out


class FakeGeminiClient:
    def __init__(self, names=(), error=None):
        self.names, self.error = names, error
        self.models = self

    def list(self):
        if self.error:
            raise self.error
        return [SimpleNamespace(name=f"models/{n}") for n in self.names]


def gemini_online(env, client):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_PROVIDER="gemini", GEMINI_API_KEY=GEMINI_KEY)
    return preflight(Settings(), online=True, gemini_factory=lambda: client)[-1]


def test_online_confirms_the_key_and_that_the_model_is_available_to_it(env):
    assert gemini_online(env, FakeGeminiClient(["gemini-3.8-flash", "gemini-3.5-flash-lite"])) == (
        OK, "Gemini accepted the key and gemini-3.8-flash is available to it.",
    )


def test_online_warns_with_alternatives_when_the_model_is_not_in_the_keys_list(env):
    level, text = gemini_online(env, FakeGeminiClient(["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "embedding-001"]))
    assert level == WARN and "gemini-3.8-flash is not in its model list" in text
    assert "Fast models it lists: gemini-3.1-flash-lite, gemini-3.5-flash-lite" in text and "SNAPNOTES_GEMINI_MODEL" in text


def test_online_reports_a_rejected_key_and_an_unreachable_service_differently(env):
    rejected = errors.APIError(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    assert gemini_online(env, FakeGeminiClient(error=rejected))[0] == BAD
    assert "rejected the key" in gemini_online(env, FakeGeminiClient(error=rejected))[1]

    level, text = gemini_online(env, FakeGeminiClient(error=ConnectionError("no network")))
    assert level == WARN and "Could not reach Gemini (ConnectionError)" in text

    server = errors.APIError(500, {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}})
    assert gemini_online(env, FakeGeminiClient(error=server))[0] == BAD


def test_online_is_skipped_in_demo_mode(env):
    env(SNAPNOTES_TOKEN=TOKEN, SNAPNOTES_DEMO="1")
    client = FakeClient(api_error(anthropic.AuthenticationError, 401))
    lines = preflight(Settings(), online=True, client_factory=lambda: client)
    assert all(level != BAD for level, _ in lines)
