# Deploying the SnapNotes backend to Fly.io

The backend is one Python process (API plus an in-process job worker) with one SQLite database on a volume. It runs as a single small machine. Phones reach it over HTTPS, which installed builds of the app require.

**Status:** the packaging was verified by installing the project into a clean virtualenv and running the full flow against it (upload, job, approval, export, sync). The same steps the `Dockerfile` performs (hash-checked install from `requirements.lock`, then the app, then the test suite) were replayed in a clean virtualenv on Python 3.14, and Linux (x86-64 and ARM) wheels exist for every locked package. The Docker image and `fly.toml` follow Fly's documentation but have not been built or deployed from this repository yet (no Docker was available), so expect to adjust a detail on the first deploy.

## You need

- A [Fly.io](https://fly.io) account (a payment card is required) and the `flyctl` CLI: <https://fly.io/docs/flyctl/install/>
- A Claude API key
- A long random access token for the app. Generate one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

  The server will not start with a token shorter than 24 characters (`SNAPNOTES_ALLOW_WEAK_TOKEN=1` overrides this, which you should not do on a public address).

## First deploy

Run everything from the `backend/` folder.

1. **Pick an app name.** Fly app names are global. Edit `app = "snapnotes-change-me"` in `fly.toml`, and set `primary_region` to a region near you (`fly platform regions`).
2. **Create the app without deploying:**
   ```bash
   fly apps create <your-app-name>
   ```
3. **Create the volume** that holds the database, in the same region as `primary_region`:
   ```bash
   fly volumes create snapnotes_data --size 1 --region <region>
   ```
   One volume means one machine. Do not scale the app to more than one machine: the database is a single file and the worker runs inside the process.
4. **Set the secrets** (they are stored by Fly and never written into `fly.toml`):
   ```bash
   fly secrets set SNAPNOTES_TOKEN=<the token you generated> ANTHROPIC_API_KEY=<your key>
   ```
   The server refuses to start without `SNAPNOTES_TOKEN`, so set it before the first deploy.
5. **Deploy:**
   ```bash
   fly deploy
   ```
6. **Check it:**
   ```bash
   curl https://<your-app-name>.fly.dev/health
   curl -H "Authorization: Bearer <token>" https://<your-app-name>.fly.dev/decks
   ```
   The first prints `{"status":"ok"}`, the second `[]`. To inspect the secrets the machine actually received (it prints their lengths, never the values):
   ```bash
   fly ssh console -C "python -m app.admin check"
   ```
7. **Connect the app:** in SnapNotes, open Settings, enter `https://<your-app-name>.fly.dev` and the token.

## Updating

```bash
fly deploy
```

The database lives on the volume, so it survives deploys. The machine restarts for a few seconds; a document being processed at that moment is re-queued automatically and runs again from the start of its phase.

The image installs the exact package versions in `requirements.lock` (each file checked against a recorded hash), so a redeploy next month runs the same code as today unless you change that file. To update dependencies deliberately, regenerate the lock files as described in the README ("Pinned dependencies"), run the tests, then deploy.

**Python version.** The image uses Python 3.14 (`ARG PYTHON_VERSION` in the `Dockerfile`), the version the test suite runs on and the lock files are resolved for. To run the tests inside the image's own environment before deploying:

```bash
docker build --target test .
```

A failing test stops that build; the normal `fly deploy` build does not run them.

## Running costs and limits

- The machine is kept running on purpose (`auto_stop_machines = "off"`). If Fly stopped it while a document was being processed, that job would restart and its AI cost would be paid twice. With 1 GB of memory this is a few dollars a month; check Fly's current pricing.
- **Memory:** preparing one maximum-size (30 MB) PDF takes about 150 MB on top of the ~120 MB the idle server uses, which is why `fly.toml` asks for 1 GB and the worker handles one document at a time (`SNAPNOTES_JOB_CONCURRENCY=1`; further uploads queue). If you drop to a 512 MB machine, also lower `SNAPNOTES_MAX_UPLOAD_MB`.
- Claude usage is the larger cost. Limits that protect it are set in `fly.toml`: at most `SNAPNOTES_MAX_JOBS_PER_HOUR` uploads per hour (10) and `SNAPNOTES_MAX_JOB_COST_USD` (2.0) of estimated spend per document, checked between stages. Change them in the `[env]` section and redeploy. The estimate excludes web search fees.
- Request sizes are capped at the edge: an upload may be at most `SNAPNOTES_MAX_UPLOAD_MB` (30) plus 1 MB of framing, anything else at most `SNAPNOTES_MAX_BODY_KB` (2048). Larger requests get `413` without being read.
- Per-client rate limiting uses Fly's `Fly-Client-IP` header (`SNAPNOTES_CLIENT_IP_HEADER`, the default). It is only trustworthy behind Fly's proxy; if you run the image anywhere else, set that variable to an empty value.
- `SNAPNOTES_DOCS=0` hides the interactive API docs on the public address.

## Using the free Gemini tier on Fly (optional)

To run the deployed server on Google's free tier instead of Claude, set `SNAPNOTES_PROVIDER = "gemini"` in the `[env]` section of `fly.toml` and store the key as a secret: `fly secrets set GEMINI_API_KEY=<your key>` (an `ANTHROPIC_API_KEY` is then not needed). Cards come from the notes alone, and free-tier content may be used by Google to improve its products, so keep private notes on Claude. `fly ssh console -C "python -m app.admin check"` shows what the machine received. See the backend README ("Free AI: Gemini instead of Claude"). Not tried on Fly yet.

## Backups

Fly takes a daily snapshot of the volume and keeps it for 5 days by default (`fly volumes snapshots list <volume id>`). For a copy you can keep yourself, make a consistent copy of the database (not a raw file copy, because it is in use) and download it:

```bash
fly ssh console -C "python -c \"import sqlite3; s = sqlite3.connect('/data/snapnotes.db'); d = sqlite3.connect('/data/backup.db'); s.backup(d)\""
fly ssh sftp get /data/backup.db ./snapnotes-backup.db
fly ssh console -C "rm /data/backup.db"
```

## Restoring a backup

Phones remember which database they last synced with. After you restore a snapshot (or a copy of `snapnotes.db`) the data is older than what the phones have seen, so tell the phones to start over:

```bash
fly ssh console -C "python -m app.admin rotate-epoch"
```

The next time each phone connects it discards its cached decks and cards, downloads everything from the restored server, and uploads its whole review history again. Study progress made after the backup was taken is restored from the phone, and cards that exist only on the phone (because they were created after the backup) are dropped. Without this step a phone could keep cards the restored server has never heard of, and reviews of them would be lost.

Recreating the volume from scratch needs no extra step: a new database gets a new identity and the phones notice on their own.

## Privacy

Uploaded notes are sent to Anthropic's API to generate cards, and the web search step queries public sources. Uploaded originals are deleted from the volume once cards are generated (`SNAPNOTES_DELETE_ORIGINALS=0` keeps them). Cards, schedules and review history stay in the database on your volume.

## Running the image locally

```bash
docker build -t snapnotes .
docker run --rm -p 8000:8000 -v snapnotes-data:/data -e SNAPNOTES_TOKEN=dev-token -e SNAPNOTES_DEMO=1 snapnotes
```

`SNAPNOTES_DEMO=1` returns canned cards, so no Claude key is needed to try it.

## Troubleshooting

- **Deploy loops or the machine will not start:** `fly logs`. A missing `SNAPNOTES_TOKEN` or an unattached volume are the usual causes.
- **The app says it cannot connect:** installed builds only talk to `https://` addresses. Use the `fly.dev` address, not `http://`.
- **Jobs fail with "no Claude credentials":** the `ANTHROPIC_API_KEY` secret is missing.
- **Uploads rejected with 429:** the hourly job limit was reached; wait, or raise `SNAPNOTES_MAX_JOBS_PER_HOUR`.
