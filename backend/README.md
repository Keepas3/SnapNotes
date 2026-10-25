# SnapNotes backend

Python pipeline that turns notes (PDF, DOCX, PPTX, Markdown/text, photos) plus web research into flashcards.
See the project plan for the full design; this folder currently holds **Milestone 0: the spike**.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

That installs the newest versions `pyproject.toml` allows, which is fine for development. The Docker image does not do that: it installs the exact, hash-checked versions in `requirements.lock` (see [Pinned dependencies](#pinned-dependencies)).

## Run the spike

Set your Claude credentials in your own shell (never commit them), then run it on a notes file:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
.\.venv\Scripts\python.exe spike.py path\to\notes.pdf
```

Useful variants:

- `spike.py page1.jpg page2.jpg` treats several photos as one document.
- `--interactive` lets you reject web sources before cards are generated.
- Results are saved to `spike_out/<name>.json`.

The run prints the cards (flagged UNGROUNDED / NOT ATOMIC / DUPLICATE where verification found problems) and the token usage with an estimated cost per stage.

**Spend limit.** The spike stops between stages once its estimated spend passes `SNAPNOTES_MAX_JOB_COST_USD` (default 2.00, the same setting the server uses per document), prints what was spent so far and exits with code 1. For a first run use a low value, for example `$env:SNAPNOTES_MAX_JOB_COST_USD = "0.50"`. The limit is checked only before a stage starts and the estimate leaves out web search fees, so a single stage can overshoot it; set a monthly spend limit in the Anthropic Console as the hard backstop.

## Run the API server (Milestone 1)

The server needs a personal bearer token and, for card generation, Claude credentials. Generate a token and set both in your own shell:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:SNAPNOTES_TOKEN = "<the string it printed>"
$env:ANTHROPIC_API_KEY = "<your key>"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

**Check the settings before you start.** Environment variables only exist in the window where you set them, and a typo in one is easy to miss. In that same window run:

```powershell
.\.venv\Scripts\python.exe -m app.admin check
```

It reads the variables the server would read and prints one line each, marked `[ok]`, `[!!]` (warning) or `[XX]` (problem): the token's length, whether an API key is present and looks like one, the per-document spend limit, whether every model has a price, and where the data will go. It never prints the token or the key, and it changes nothing on disk. It catches the usual mistakes, such as leaving the `<placeholder>` brackets or quotes in a value, a token that is too short, or a key set in a different window. Add `--online` to also ask Claude whether it accepts the key; that call is free and uses no tokens. It exits with code 1 when there is a problem.

Interactive docs are at http://127.0.0.1:8000/docs (click Authorize and paste the token). For local experiments without a token, set `SNAPNOTES_ALLOW_NO_AUTH=1` instead. The server refuses to start with neither.

**The token must be at least 24 characters.** The server is reachable from the internet once deployed and the token is the only thing protecting your library and your Claude bill, so a short one such as `dev-token` is refused with a message that includes the command above. Demo mode (below) is exempt because it spends nothing; `SNAPNOTES_ALLOW_WEAK_TOKEN=1` switches the check off on purpose for anything else.

Data lives in `data/` (SQLite database plus uploaded originals); change it with `SNAPNOTES_DATA_DIR`.

### Demo mode: develop the app without a Claude key

`SNAPNOTES_DEMO=1` swaps Claude for canned responses (a small cell-biology deck, including one duplicate and one unsupported card so the review screens have something to show). The real worker, database, source-approval step and API all run unchanged, and nothing is spent.

```powershell
$env:SNAPNOTES_DEMO = "1"
$env:SNAPNOTES_TOKEN = "dev-token"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --host 0.0.0.0
```

`SNAPNOTES_DEMO_DELAY` (seconds per fake Claude call, default 1.2) controls how long the progress screen stays visible. Use `--host 0.0.0.0` to reach the server from a phone on your Wi-Fi.

For the Expo **web** preview only, also set `SNAPNOTES_CORS_ORIGINS=http://localhost:8081` so the browser may call the API. Native apps do not need CORS, and it is off by default.

### Free AI: Gemini instead of Claude

For testing the app without paying for Claude, one setting switches the whole server to Google's free Gemini tier.

| | Claude (default) | Gemini (free tier) |
|---|---|---|
| Cost | per document (see the spend limits below) | none; Google limits requests per minute and per day |
| Web research and source approval | yes | **no**: every card comes from your notes alone |
| PDFs and photos | read natively | sent to the model as they are (not yet tried with a real key) |
| Your notes | sent to Anthropic | sent to Google, which **may use free-tier content to improve its products** (its pricing page says so), so test with notes that are not private |

Setup (in the window that will run the server):

1. Make a free key in [Google AI Studio](https://aistudio.google.com) ("Get API key").
2. Set the variables and check them (the key is never printed):

```powershell
$env:SNAPNOTES_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "<your free key>"
.\.venv\Scripts\python.exe -m app.admin check --online
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

`check --online` asks Google for the key's model list (free) and tells you whether the configured model, `gemini-3.8-flash` unless you set `SNAPNOTES_GEMINI_MODEL`, is one the key can use. Which models are free changes over time; pick another from that list if needed. `GOOGLE_API_KEY` works instead of `GEMINI_API_KEY`. To go back to Claude, remove `SNAPNOTES_PROVIDER` (or set it to `anthropic`) and restart.

The per-document spend limit does not apply to the free tier (it costs nothing); Google's own limits do. When Google says you are over them the job fails with "The free AI is rate limiting requests, or today's free allowance is used up", and you can retry it a minute later or the next day. The same switch works for `spike.py`.

**Which free model.** On the same 385-byte sample, `gemini-3.8-flash` took about 13 minutes (three calls of roughly 1,000 tokens each; Google's free tier was simply slow) while `gemini-3.5-flash-lite` took 8 seconds. The lighter model was less careful: it turned the sample's deliberately wrong line ("ribosomes are made of protein only") into a card, because the notes say it and, with no web research, nothing checks the notes against the real world. The slower one did not. For testing the app's flow, set `SNAPNOTES_GEMINI_MODEL=gemini-3.5-flash-lite`; read the cards before trusting them. Timings vary with how busy Google is, and the server window prints a warning for any call over 30 seconds.

**Fact check (on by default with Gemini).** With no web research, nothing else compares your notes with the real world, so the same two calls that already run also ask the model about its own general knowledge: the first step lists statements in your notes that are clearly wrong, each with a one-line correction (`suspect_claims`), and the quality check can add a "May be wrong: ..." remark to a card. A statement on the list gets **no card made from it as written**; the app shows the list on the job screen and on the approve-sources screen. A card with a remark stays a draft with an amber warning in the review, so you decide. Limits: it is the model's memory, not a source, so it misses errors and sometimes doubts a fair simplification (for the sample's "only from the mother" it may say rare paternal inheritance has been reported). `SNAPNOTES_FACT_CHECK` is `auto` (on for Gemini, off for Claude, whose requests stay exactly as they were), `1` or `0`; `python -m app.admin check` says which. Try it with no key at all: `SNAPNOTES_DEMO=1` and `SNAPNOTES_FACT_CHECK=1`.

**Corrected cards.** Listing a wrong statement is not the end: the app then tries to teach the right fact.

1. **Search, only when needed.** If the fact check found statements that look wrong, the free path asks Gemini's built-in Google Search to confirm or correct each one. That sends the topic and the doubtful statements (not your notes) to Google, takes roughly 10 to 30 seconds more, and counts against Google's free search allowance (their pricing page lists 5,000 searches a month for Gemini 3 and newer; your own limit is shown in Google AI Studio). A document with nothing doubtful makes no search at all. Sources it finds go through the usual "approve sources" screen. The links are Google redirect links, and what it returns is the model's own supported sentence, not the passage it read.
2. **A card from a source.** If a source settles a statement, the card is written from it ("From the web").
3. **A card from the AI's own knowledge.** If nothing settles it and the model is certain of the right fact, it writes one card, labelled **"From AI knowledge, not sourced"**, with a warning, and it stays a draft until you approve it. The quality check gives a second opinion on it. `SNAPNOTES_AI_CORRECTIONS=0` turns this off (sourced corrections only).
4. **If the search fails** (limit reached, network, Google refusing the tool for your key or model), the server prints a warning, the job carries on without sources, and any correction comes from step 3. The app says so too, under the list of doubtful statements ("The web search did not work (...), so the corrections come from the AI's own knowledge").

To see exactly what Google does with a search from your key, without uploading anything, run `python -m app.admin check --search`. It makes one real search with a harmless question (one of your free searches) and prints either the sources it found or Google's own reason for refusing.

The same research change applies to Claude when the fact check is switched on (the statements are added to its research request). Claude's web research has not yet been shown to find sources on a real run, so that half is untested. Anki exports tag these cards `source-ai` and say "From AI knowledge (not sourced)" on the card; the CSV `origin` column says `ai`.

What this does not test: Claude's web search and citations, prompt caching and cost numbers. Use Claude for those.

### How a document becomes cards

1. `POST /documents` (multipart `files`, optional `auto_approve=true`, optional `deck_id`) returns a `job_id` immediately. With a `deck_id` the cards are added to that existing deck instead of a new one (see "Growing a deck" below); an unknown deck is a 404 before anything is saved.
2. Poll `GET /jobs/{id}`. Status goes `queued` -> `researching` -> `awaiting_approval`.
3. `GET /jobs/{id}/evidence` lists the web sources found. `POST /jobs/{id}/approve` with `{"rejected_ids": [...]}` approves the rest. (With no web evidence, or `auto_approve`, this step is skipped.)
4. Status goes `approved` -> `generating` -> `completed`, and `deck_id` is set.
5. `GET /jobs/{id}/cards` returns draft cards. Cards that failed verification or are duplicates start as `rejected`. Approve or reject in bulk with `POST /cards/review`, or edit one with `PATCH /cards/{id}`.

A `failed` job has a readable `error`; `POST /jobs/{id}/retry` re-runs it from the phase that failed. Uploaded originals are deleted once cards exist (`SNAPNOTES_DELETE_ORIGINALS=0` to keep them), or on demand with `DELETE /documents/{id}`, which also removes every card that was never approved.

### Growing a deck

Three ways to add to a deck you already have:

- **Add notes:** upload with `deck_id` (above). When the job finishes, its cards are drafts in that deck, the deck's name and topic are left alone, and `jobs.deck_id` points at it (`target_deck_id` on the job says it was asked to). A new card that repeats a **draft or approved** card already in the deck is stored as `rejected` with `duplicate_of` set to the existing card and the issue "Already in this deck." The match is the same strict rule used inside one upload (normalised fronts, `token_sort_ratio >= 95`), so a *reworded* question about the same fact is not caught. If the deck is deleted while the job is running, the job makes a new deck as usual.
- **Merge:** `POST /decks/{id}/merge` with `{"from_deck_id": "..."}` moves every card of the other deck (any status) into this one and removes the other deck, in one transaction. Cards keep their ids, so review logs and schedules carry over untouched. Uploads that were headed for, or came from, the removed deck are re-pointed at this one. Merging a deck into itself is a 409, an unknown deck a 404. The response is `{"deck": ..., "moved": n}`.
- **Type a card:** `POST /decks/{id}/cards` with `{"type": "basic" | "cloze", "front": "...", "back": "..."}` creates an **approved** card at once (nothing to review): `origin` is `manual`, `source_ref` is "Added by you", and it has no job. A basic card needs both sides; a cloze card needs a `{{c1::blank}}` in the front and may have an empty back. Problems come back as a 422 with a sentence. Anki exports tag these cards `source-manual` and say "Added by you"; the CSV `origin` column says `manual`.

### Folders

Folders group decks that share a subject (Biology: Bio I, Bio II). One level only: a folder holds decks, never other folders. A deck is in at most one folder, and a folder never owns its decks.

- `GET /folders`, `POST /folders {"name": "..."}`, `PATCH /folders/{id} {"name": "..."}`, `DELETE /folders/{id}`. Names are 1 to 80 characters and unique ignoring case (409 with a sentence otherwise). Each folder reports its `deck_count`.
- **Filing a deck:** `PATCH /decks/{id}` with `{"folder_id": "<folder id>"}` files it, `{"folder_id": null}` takes it out, and leaving `folder_id` out changes nothing about it. `DeckOut` carries `folder_id`. An unknown folder is a 404.
- **Deleting a folder never deletes decks:** they are taken out of it first (each deck's change is recorded, so phones hear about it), then the folder goes. Merging decks leaves the destination in its own folder.
- **Sync:** a pull also returns `folders`, every deck carries its `folder_id`, and `deleted` can name a `folder`. Pages still end at one cursor for every kind of row, so a phone never learns a folder is gone before it has learned its decks left it. A phone that predates folders ignores the new fields.

### Export

`GET /decks/{id}/export?format=apkg|csv` downloads the deck's **approved** cards (409 if there are none).

- **`apkg`** is an Anki package built with `genanki`: a "SnapNotes Basic" note type (front, back, source) and a "SnapNotes Cloze" note type, using Anki's own `{{c1::...}}` cloze syntax. Tags are cleaned of spaces and marked `snapnotes` plus `source-notes` or `source-web`; web sources become links only if they are plain `http(s)` URLs. Note ids and note-type ids are stable, so importing a re-export updates the existing notes. A "cloze" card whose blanks were edited away is exported as a basic card (Anki would otherwise create a note with no cards). Cards arrive as new; review history is not exported.
- **`csv`** has the columns `deck, type, front, back, source, origin, tags, difficulty`, UTF-8 with a byte-order mark, and a leading `'` on any cell that starts with `=`, `+`, `-` or `@` so spreadsheets cannot execute note text as a formula.

### Deploying

`DEPLOY.md` walks through Docker + Fly.io (`Dockerfile`, `fly.toml`).

### Sync (Milestone 3)

`POST /sync` is how the phone studies offline and still stays in step with the server:

```json
{ "device_id": "...", "cursor": 0, "limit": 500, "logs": [ { "id": "...", "card_id": "...", "rating": 3, "reviewed_at": "2026-03-01T09:00:00.000Z",
    "elapsed_ms": 2400, "state_before": 0, "state_after": { "due": "...", "stability": 4.2, "difficulty": 5.1, "reps": 1, "lapses": 0, "state": 2, "last_review": null } } ] }
```

The response carries `decks`, `cards` (with their `status`, so the device can drop cards that are no longer approved), `states` (the latest FSRS schedule per card), `deleted` (tombstones), a new `cursor`, `has_more`, and the database's `epoch` and `reset` flag (below). Keep calling with the returned cursor and epoch while `has_more` is true.

- **Pull is cursor based.** A global change counter (`sync_counter`) is stamped onto every changed deck, card and schedule by database triggers, so no write path can forget it; deletions, including cascades, leave tombstones. Losing a change is never allowed. A page that is cut short ends at one cursor for every kind of row, so changes always arrive in the order they happened: a merged deck's cards are moved before the deck is deleted, and a device drops a deck's cards when told the deck is gone, so that deletion must not arrive before the moves.
- **Push is idempotent.** Review logs are keyed by a client-generated id, so a retried request never duplicates a review. A bad log (unknown card, rating outside 1-4, a date more than a day in the future) is listed in `rejected` and the rest of the batch is still applied.
- **The device computes the schedule.** Each log carries the FSRS state after the review; the server stores the log and keeps the state of the most recent review per card (an older log never rewinds it). Cards always arrive together with their schedule.
- **A device can tell when it is no longer talking to the database it synced with.** Every database has a random `epoch` created with it, sent back by the device on each request. If the epoch differs, or the device's cursor is ahead of anything this database has ever numbered (the server was rolled back), the response has `reset: true` and is a full download from zero. The device then discards its cached decks and cards (including cards the server no longer has), keeps its review history, and uploads all of it again: reviews the server lost are restored, reviews it still has are ignored as duplicates, and reviews of cards that no longer exist are rejected. This covers a rebuilt volume automatically. A restored snapshot has the *old* epoch, so after a restore run `python -m app.admin rotate-epoch` (see `DEPLOY.md`).
- **Existing databases upgrade in place** (schema version 5, tracked with `PRAGMA user_version`; version 4 added `jobs.target_deck_id`, version 5 added folders and `decks.folder_id`); rows that predate sync are numbered oldest first.

### Spend and safety limits

| Setting | Default | Effect |
|---|---|---|
| `SNAPNOTES_MAX_JOBS_PER_HOUR` | 10 | `429` on upload beyond this |
| `SNAPNOTES_MAX_JOB_COST_USD` | 2.0 | job stops between stages once its estimated spend passes this |
| `SNAPNOTES_REQUESTS_PER_MINUTE` | 240 | per-client request limit (`429`) |
| `SNAPNOTES_CLIENT_IP_HEADER` | `fly-client-ip` | request header that names the client for the rate limit (set by Fly's proxy). Set it to an empty value anywhere else, so a caller cannot dodge the limit by sending the header themselves; the connection's own address is used instead |
| `SNAPNOTES_JOB_CONCURRENCY` | 1 | documents processed at once (others queue). Preparing a maximum-size PDF takes ~150 MB, so raise this only on a larger machine |
| `SNAPNOTES_MAX_UPLOAD_MB` | 30 | largest upload accepted (also the Claude PDF limit); lower it on a small machine |
| `SNAPNOTES_MAX_BODY_KB` | 2048 | largest request body for everything except `POST /documents` (a full sync batch is far below this) |
| `SNAPNOTES_ALLOW_WEAK_TOKEN` | off | set to 1 to allow an access token shorter than 24 characters (see above) |
| `SNAPNOTES_EXTRA_PRICES` | (none) | `{"model-id": [input_usd_per_1m_tokens, output_usd_per_1m_tokens]}` for models not in the built-in price table |
| `SNAPNOTES_DOCS` | 1 | set to 0 to hide `/docs` on a public host |
| `SNAPNOTES_PROVIDER` | `anthropic` | `gemini` switches the whole server to Google's free tier (see "Free AI" above) |
| `SNAPNOTES_GEMINI_MODEL` | `gemini-3.8-flash` | the model used for every stage when the provider is `gemini` |
| `SNAPNOTES_AI_CORRECTIONS` | `1` | when no source settles a statement the fact check found wrong, still make one card with the right fact from the AI's own knowledge, labelled and held as a draft (`0` = sourced corrections only) |
| `SNAPNOTES_FACT_CHECK` | `auto` | list statements in the notes that look factually wrong and flag doubtful cards: `auto` = on for `gemini`, off for `anthropic`; `1` or `0` to force |

The cost estimate excludes web search fees, and a single stage can overshoot the limit because it is only checked between stages.

**Oversized requests are refused before they are read.** A small middleware (`app/core/bodylimit.py`) answers `413` as soon as a request announces a body above the limit, without receiving any of it, and cuts off a body that arrives without a length (chunked) once it passes the limit. Uploads get `SNAPNOTES_MAX_UPLOAD_MB` plus 1 MB for the multipart wrapping; every other endpoint gets `SNAPNOTES_MAX_BODY_KB`. The handler still checks the total size of an upload's files itself.

**The rate limiter has a bounded memory.** It remembers at most 10,000 clients (the least recently seen is forgotten first) and sweeps out expired ones as it goes, so a flood of made-up client addresses cannot grow it without limit.

**The spend limit fails closed.** It needs a price for every model in use. The built-in table covers `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-haiku-5-5`; if a stage is pointed at any other model, the server refuses to start and prints the `SNAPNOTES_EXTRA_PRICES` setting to add. (A job whose cost still cannot be estimated is stopped rather than allowed to spend unchecked.)

## Tuning

Per-stage model and effort live in `app/core/config.py`. Override a stage's model with an environment variable, for example:

```powershell
$env:SNAPNOTES_VERIFY_MODEL = "claude-sonnet-5-5"
```

**When web research finds nothing.** If the search step runs but no evidence is collected, the server prints a warning (in the window running it) naming the search count and the kinds of content and citations the API returned. For the full picture set `SNAPNOTES_DEBUG_DUMP` to a folder (for example `$env:SNAPNOTES_DEBUG_DUMP = "debug-dump"`), restart the server and run a document again: every raw research response is saved there as a JSON file (page text from the web, so keep it local; the folder is git-ignored and not part of the image). It is off by default.

Other switches: `SNAPNOTES_FALLBACKS=0`, `SNAPNOTES_WEB_MAX_USES`, `SNAPNOTES_WEB_ALLOWED_DOMAINS` / `SNAPNOTES_WEB_BLOCKED_DOMAINS` (comma separated, use one), `SNAPNOTES_MAX_CARDS`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use a faked SDK, so they run offline and cost nothing. To run them inside the deployment image's own environment (same Python, same locked packages) without starting anything:

```bash
docker build --target test .
```

(One test, which checks `fly.toml` against `DEPLOY.md`, is skipped there because neither file is part of the Docker build context.)

## Pinned dependencies

`pyproject.toml` only states what the code needs (lower bounds). The versions that were actually tested are recorded, with a checksum for every file, in two lock files:

- `requirements.lock`: the server's runtime packages. The Dockerfile installs exactly this (`pip install --require-hashes --no-deps`), so two builds months apart contain the same bytes, and a package that has been swapped or tampered with on the index is refused.
- `requirements-dev.lock`: the same plus the test tools (`pytest`, `httpx`); used by the Docker test stage.

Both are resolved for Python 3.14, the version the image uses (`ARG PYTHON_VERSION` in the Dockerfile) and the one the tests run on. If you change a dependency in `pyproject.toml` or want newer versions, regenerate them, run the tests, and commit the new files together:

```powershell
python -m pip install uv
uv pip compile pyproject.toml --universal --python-version 3.14 --generate-hashes -o requirements.lock
uv pip compile pyproject.toml --universal --python-version 3.14 --generate-hashes --extra dev -o requirements-dev.lock
```

To move to another Python, change the `--python-version` above and `PYTHON_VERSION` in the Dockerfile together. `pip-audit -r requirements.lock` checks the pinned set for known vulnerabilities.

The one thing not pinned is the build tool for the app's own package (`setuptools`, named in `pyproject.toml`); it is used during the image build and is not part of the running image.

## Layout

```
app/main.py            FastAPI app factory (create_app), rate limit middleware, lifespan
app/api/               routes: uploads, jobs, decks, cards, sync (+ request/response models)
app/core/              config, auth (bearer token), rate limiter, request size limit, SQLite access, schema and migrations
app/store/             JobStore (documents, jobs, evidence), Library (decks, cards), SyncStore (change feed, review logs)
app/export/            Anki .apkg (genanki) and CSV export
Dockerfile, fly.toml   container image and Fly.io configuration (see DEPLOY.md)
requirements*.lock     pinned, hash-checked dependencies used by the image (see "Pinned dependencies")
app/ingest/loader.py   files -> Claude content blocks, upload-time validation
app/llm/               Claude client (client.py), Gemini client (gemini.py), the shared interface (base.py) and the choice between them (factory.py); schemas, prompts, understand / enrich / generate / verify
app/pipeline/          orchestrator (research / generation phases) and the background job worker
spike.py               command-line runner
tests/                 offline tests (faked Claude SDK; API tests use FastAPI's TestClient)
```
