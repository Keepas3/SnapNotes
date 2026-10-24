# SnapNotes

Turn your notes into flashcards. Upload a PDF, slides, a document or photos of paper notes; SnapNotes works out what they cover, finds trustworthy web sources to fill gaps (you approve them), and writes flashcards that say whether each fact came from your notes or the web. You review the cards, then study them with spaced repetition, offline.

| Folder | What it is |
|---|---|
| [`backend/`](backend/) | Python (FastAPI + SQLite). Reads documents, calls Claude to understand them and search the web, generates and checks cards, stores the library, syncs devices, and exports to Anki and CSV |
| [`mobile/`](mobile/) | Expo / React Native app (iOS and Android). Upload and camera capture, source approval, swipe review, and offline FSRS study on an on-device database |

## Try it in five minutes (no Claude key, no cost)

Demo mode returns canned cards, so you can run the whole app without an API key.

```powershell
# 1. backend
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:SNAPNOTES_DEMO = "1"; $env:SNAPNOTES_TOKEN = "dev-token"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# 2. app (in another terminal)
cd mobile
npm install
npx expo start      # scan the QR code with Expo Go, or press w for the browser preview
```

In the app's Settings enter the server address (`http://<your computer's Wi-Fi address>:8000` on a phone, `http://localhost:8000` in the browser preview) and the token `dev-token`. Details for both halves are in [`backend/README.md`](backend/README.md) and [`mobile/README.md`](mobile/README.md).

## Using it for real

1. **Deploy the backend** with a Claude API key: [`backend/DEPLOY.md`](backend/DEPLOY.md) (Docker + Fly.io, HTTPS).
2. **Install the app** on your phone: [Installable builds](mobile/README.md#installable-builds-eas) (EAS).
3. In the app's Settings, enter your `https://` server address and token.

## How it fits together

```
 phone (Expo app)                              server (FastAPI)                          Claude API
 pick file / photo ──── upload ───────────────▶ read + understand the notes ────────────▶ vision / PDF input
 approve web sources ◀─ evidence to approve ─── web search with citations ◀─────────────── web search tool
 swipe to review ◀───── draft cards ─────────── write + check cards (grounded, atomic, no duplicates)
 study offline (FSRS, on-device SQLite)
 sync reviews ────────▶ review logs + schedules; changes come back  ·  export: Anki .apkg, CSV
```

- Every card carries its origin (notes or web), a source reference, and a supporting snippet.
- Notes and web pages are treated as data, never as instructions.
- Spend is bounded: an hourly job limit and a per-document cost ceiling.

## Checks

```bash
cd backend && .venv/Scripts/python -m pytest -q        # 276 tests, offline (Claude and Gemini are faked)
cd mobile  && npm run typecheck && npm run lint && npm test   # 136 tests, on real SQLite
```

## Trying it with a free AI

Claude costs money per document. For testing, the server can use the Gemini free tier instead (`SNAPNOTES_PROVIDER=gemini`, key in `GEMINI_API_KEY`): see [Free AI: Gemini instead of Claude](backend/README.md#free-ai-gemini-instead-of-claude). It reads your notes and writes cards, checks them against general knowledge and flags statements that look wrong. Its Google Search step for making corrected cards is built and unit-tested but has never returned a result against the real service (the free key hit its quota), so corrections come from the model's own knowledge and are labelled as unsourced. Google may use free-tier content to improve its products, so test with notes you do not mind sharing.

## Status

This is a personal project in early development. What has and has not been exercised:

- The backend and the app are covered by the tests above, and both ran end to end on an Android emulator against the free Gemini tier and in demo mode.
- The Claude path has been through the real API only once, on one short document. Web research returned no sources in that run and the cause is not yet known, so treat Claude web research as unproven. The cost figures it printed are the only real ones so far.
- The Docker image, the Fly.io deployment and the EAS builds have never been run.
- Your notes are sent to the AI provider you choose (Anthropic, or Google for the free path). Read the provider's terms before using private notes, and see [`REVIEW.md`](REVIEW.md) for the known issues and what was checked.

## Secrets

Nothing in this repository is a credential. API keys and the access token are read from environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `SNAPNOTES_TOKEN`), and the `dev-token` in the demo instructions only works in demo mode, which returns canned data and spends nothing. `.env` files, databases and local tool settings are git-ignored. `python -m app.admin check` tells you which settings the server can see without printing them.

## License

[MIT](LICENSE). The app in `mobile/` started from the Expo project template, whose own MIT notice (copyright Expo) is kept in [`mobile/LICENSE`](mobile/LICENSE) for the template files that remain, such as the default icons.
