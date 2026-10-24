# SnapNotes mobile app

Expo (SDK 58) + React Native + TypeScript, using Expo Router. It talks to the backend in `../backend`.

## What is in it

| Screen | File | What it does |
|---|---|---|
| Home | `src/app/index.tsx` | Decks (from the on-device database, so it works offline) with due / new counts and a Study button, "to review" badges, jobs in progress, and sync status |
| Study | `src/app/study/[deckId].tsx` | Spaced-repetition session for one deck or all decks (`all`): show answer, then Again / Hard / Good / Easy with the next interval on each button. Runs entirely on the device |
| New deck | `src/app/new.tsx` | Take a photo, choose photos, or choose PDF / Word / slides / text; optional "skip source approval" |
| Job progress | `src/app/jobs/[id]/index.tsx` | Five-step progress, retry or discard, and moves you on when the job needs you or finishes |
| Approve sources | `src/app/jobs/[id]/sources.tsx` | Switch web sources on or off before cards are written; links open only as http(s) |
| Review cards | `src/app/jobs/[id]/review.tsx` | Swipe right to approve, left to discard, tap to flip, edit, undo; cards the checker set aside can be restored |
| Deck | `src/app/decks/[id].tsx` | Approved / to review / set aside, edit or delete cards, and Study |
| Settings | `src/app/settings.tsx` | Server address and access token (stored in the OS keychain) with a connection test, new cards per day, sync status, disconnect |

Shared code: `src/api` (typed client, TanStack Query hooks), `src/lib` (settings, helpers for cloze cards, job steps, swipe thresholds, upload rules), `src/components` (UI kit, swipe card, card editor), `src/db` (on-device SQLite), `src/study` (FSRS scheduler and local store), `src/sync` (sync engine and provider).

## Studying offline

The phone keeps its own SQLite database: every deck, and the **approved** cards with their FSRS schedule. Studying only reads and writes that database, so it needs no connection.

- **Scheduling:** [FSRS](https://github.com/open-spaced-repetition/ts-fsrs) with default parameters and 90% target retention, fuzzing off (the same history always gives the same schedule). New cards go through learning steps (1 min, 10 min), then are scheduled in days.
- **Answering:** tap the card to see the answer, then swipe it **right if you knew it** or **left if you missed it**; a swipe is a shortcut for the matching button. The four buttons below say how well you knew it, in plain words, and each shows when the card will come back: **Missed it** (FSRS "Again"), **Hard**, **Got it** ("Good", what a right swipe records) and **Easy**. The better you knew it, the longer the wait. The card does not move until the answer is showing.
- **Daily plan:** learning cards that are due, then reviews due today, then new cards up to the daily limit (Settings, default 10, shared across decks), then learning cards due within 20 minutes. A study day rolls over at 4 a.m. local time.
- **Sync:** every answer is stored as an append-only review log with a client-generated id. A sync uploads pending logs and pulls changes in one request (`POST /sync`), repeating while the server says there is more. Logs are marked as uploaded only after the response has been applied, in the same transaction, so a failed or interrupted sync just resends them and the server ignores duplicates.
- **When it runs:** on launch, when the app returns to the foreground, when Home or a deck comes into view, after a study session, and from "Sync now".
- **Conflicts:** the most recent review of a card wins its schedule. If two devices review the same card while offline, both reviews are kept in the history, but only the later one sets the schedule.
- **Switching servers** clears the cached library (the cards belong to the old server); you are warned first if reviews have not been uploaded yet.
- **A server that was restored or rebuilt** is noticed automatically: each response names the server's database, and if it is not the one this phone last synced with the server asks for a reset. The phone then replaces its cached decks and cards with a fresh download (so nothing the server no longer has stays studyable), keeps its review history, and uploads all of it again, which restores any progress the server lost. See `backend/DEPLOY.md` for the one command to run after restoring a backup.

On the web preview the database is sql.js (SQLite compiled to JavaScript) persisted in `localStorage`; native builds use `expo-sqlite`. The SQL is the same, and the tests run it on Node's built-in SQLite.

## Run it

```bash
npm install
npx expo start
```

Open it with Expo Go on your phone (scan the QR code), or press `w` for the web preview. Native features such as the camera need a real device or emulator; the web preview is for quick UI work.

**Android emulator:** start the emulator first, then run plain `npx expo start` and press `a`. Expo needs `ANDROID_HOME` set to find the emulator, for example `$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"` in PowerShell.

**Do not use `--localhost` on Windows with a recent Node.** The dev server then listens only on IPv6 (`::1`), while the `adb reverse` tunnel that Expo sets up connects over IPv4 and is refused, and Expo Go fails with "Something went wrong" (its log says "Failed to download remote update"). Seen with Node 24 on Windows 11; the default mode listens on both and works.

If Expo Go sits on its "SnapNotes" screen with "New update available, downloading...", wait a minute or two first: the first bundle of a session takes a while to build. Once it stayed there for several minutes while the dev server was healthy, and opening the same server as `exp://127.0.0.1:8081` loaded the app at once (`adb shell am start -a android.intent.action.VIEW -d exp://127.0.0.1:8081`, which only works while the dev server is in its default mode). The cause was never established, and the app later loaded fine through the Wi-Fi address too, so it may simply have been the first-time build.

### Connect it to a backend

1. Start the backend (see `../backend/README.md`). For development without a Claude key use demo mode.
2. In the app's Settings enter the server address and token.
   - Physical phone: `http://<your computer's Wi-Fi address>:8000` (start the server with `--host 0.0.0.0`; allow it through the Windows firewall).
   - Android emulator: `http://10.0.2.2:8000`.
   - Web preview: `http://localhost:8000`, and start the backend with `SNAPNOTES_CORS_ORIGINS=http://localhost:8081`.

Plain `http://` works in Expo Go and the web preview. Installed builds (Milestone 4) will need the backend behind HTTPS.

## Managing decks

Each deck on the home screen has two small buttons: the **✕** at the top right deletes the deck (after an "are you sure" that says how many cards go with it; this needs a connection, because the server holds the library), and the **⚙** at the top left opens the deck's settings, where you can **rename** it. The same sheet lists "add more cards to this deck" and "merge this deck with another one" as **coming soon**; they are placeholders and do nothing yet.

## Exporting a deck

On a deck's page, **Export** sends the approved cards through the system share sheet (in the browser preview it downloads a file):

- **Anki (.apkg):** open it in AnkiDroid or AnkiMobile. Cloze cards, tags and source links come across; cards arrive as new (your study history stays in SnapNotes). Importing the same deck again updates the cards instead of duplicating them.
- **CSV:** one row per card with deck, type, front, back, source, origin, tags and difficulty, in UTF-8 for Excel and Sheets. Cells that begin with `=`, `+`, `-` or `@` get a leading `'` so spreadsheets cannot run them as formulas.

## Installable builds (EAS)

Expo Go is for development. To install SnapNotes on a phone as a normal app, build it with [EAS Build](https://docs.expo.dev/build/introduction/) using the profiles in `eas.json`:

| Profile | Result | Use for |
|---|---|---|
| `preview` | Android `.apk`, or an iOS ad-hoc build, shared by an install link | Trying it on your own devices |
| `production` | Store builds (`.aab` / `.ipa`) with an auto-incrementing build number | Google Play / App Store (`eas submit`) |

One-time setup:

1. Create a free Expo account and sign in: `npx eas-cli@latest login`.
2. Link the project: `npx eas-cli@latest init` (this adds the project id to `app.json`).
3. In `app.json`, replace the placeholder identifiers `com.snapnotes.app` (`ios.bundleIdentifier` and `android.package`) with your own reverse-domain name.
4. Deploy the backend over HTTPS ([`../backend/DEPLOY.md`](../backend/DEPLOY.md)). **Installed builds refuse plain `http://` servers**, and the Settings screen says so if you try one. Expo Go and the dev server are exempt, so keep using those against a laptop on your Wi-Fi.

Then:

```bash
npm run build:preview:android    # prints an install link / QR code when the build finishes
npm run build:preview:ios        # needs an Apple Developer account and your device registered (eas device:create)
npm run build:production         # store builds
```

EAS may ask to initialise a git repository the first time, because it packages the project from git. Building and submitting to the stores also involves accounts, signing credentials and store listings that EAS walks you through; none of that has been run from this repository yet.

## Checks

```bash
npm run typecheck   # tsc --noEmit
npm run lint        # expo lint
npm test            # jest: API client, FSRS scheduler, local store and sync engine on real SQLite, helpers
```

## Not covered yet

Anki export, installed builds and push reminders (Milestone 4). The camera, photo library, document picker and `expo-sqlite` are native modules that have not been exercised on a real device yet; their web equivalents were (the study session and offline-then-sync flow were run end to end in the browser preview against the demo backend).
