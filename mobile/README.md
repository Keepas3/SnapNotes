# SnapNotes mobile app

Expo (SDK 58) + React Native + TypeScript, using Expo Router. It talks to the backend in `../backend`.

## What is in it

| Screen | File | What it does |
|---|---|---|
| Home | `src/app/(tabs)/index.tsx` | Decks (from the on-device database, so it works offline) with due / new counts and a Study button, "to review" badges, jobs in progress, and sync status |
| Study | `src/app/study/[deckId].tsx` | Spaced-repetition session for one deck or all decks (`all`): show answer, then Missed it / Got it with the next interval on each button. Runs entirely on the device |
| New deck / add to a deck | `src/app/new.tsx` | Type or paste notes into a box (up to 20,000 characters, about 3,000 words; longer notes go in as a file), take a photo, choose photos from the camera roll, or choose PDF / Word / slides / text, in any mix; optional "skip source approval". Opened from a deck's settings it adds to that deck instead |
| Job progress | `src/app/jobs/[id]/index.tsx` | Five-step progress, retry or discard, and moves you on when the job needs you or finishes |
| Approve sources | `src/app/jobs/[id]/sources.tsx` | Switch web sources on or off before cards are written; links open only as http(s) |
| Review cards | `src/app/jobs/[id]/review.tsx` | Swipe right to approve, left to discard, tap to flip, edit, undo; cards the checker set aside can be restored |
| Deck | `src/app/decks/[id].tsx` | Approved / to review / set aside, edit or delete cards, and Study; **Settings** (rename, add notes, merge) |
| Add a card | `src/app/cards/new.tsx` | Type your own question and answer (or a fill-in-the-blank) into a deck |
| Settings | `src/app/(tabs)/settings.tsx` | Server address and access token (stored in the OS keychain) with a connection test, new cards per day, sync status, disconnect |

Shared code: `src/api` (typed client, TanStack Query hooks), `src/lib` (settings, helpers for cloze cards, job steps, swipe thresholds, upload rules), `src/components` (UI kit, swipe card, card editor), `src/db` (on-device SQLite), `src/study` (FSRS scheduler and local store), `src/sync` (sync engine and provider).

## Studying offline

The phone keeps its own SQLite database: every deck, and the **approved** cards with their FSRS schedule. Studying only reads and writes that database, so it needs no connection.

- **Scheduling:** [FSRS](https://github.com/open-spaced-repetition/ts-fsrs), the algorithm Anki now offers beside its old one, with its default parameters. New cards go through learning steps (1 min, 10 min), then are scheduled in days. Two settings (Settings → Studying) change it: **how sure to be of remembering** (80, 85, 90 or 95%, default 90; higher brings cards back sooner) and **spread out reviews** (on by default: longer waits are nudged by a day or so, seeded by the card, so cards learned together do not all return on one day and the time shown under a button is the time the card gets; waits under 2.5 days are never changed). Both apply to future answers; cards already scheduled keep their dates until you answer them, or until you press **Update scheduled cards** under the retention setting. That shows what would change (how many cards come back sooner or later, how many would be due today), then gives each card in review the date the new retention implies for the strength it already has (last review + stability × the retention's interval factor), without touching cards being learned or your answers. It can be changed back the same way. It is done on this phone only, since the server hears about schedules through answers: after a reinstall, a new phone or a server reset, press it again.
- **Answering:** tap the card to see the answer, then swipe it **right if you knew it** or **left if you missed it**; a swipe is a shortcut for the matching button. Two buttons below do the same, and each shows when the card will come back: **Missed it** (FSRS "Again") and **Got it** ("Good", what a right swipe records). There are no Hard and Easy buttons, to keep studying quick; the scheduler still understands them, so they could come back as an option. A card you get over and over waits longer each time. The card does not move until the answer is showing.
- **Daily plan:** learning cards that are due come first. Then reviews due today and new cards (up to the daily limit, Settings, default 10, shared across decks) are **mixed**, with a new card chosen in proportion to how many new cards and reviews are left (the switch **Mix new cards in with reviews**, on by default; off shows all reviews first). **New card order** is **Shuffled** (default: a random-looking order that mixes decks and topics, the same all day and different tomorrow) or **In order** (the order the cards were made). Last, learning cards due within 20 minutes. A study day rolls over at 4 a.m. local time. Among reviews, the card **least likely to be remembered right now** comes first (it weighs how long ago you saw it against how strongly you know it), not simply the oldest due date; the 300 most overdue are weighed. There is no daily cap on reviews.
- **Undo:** while the study screen is open, **Undo last answer** takes back the most recent answer, then the one before it, back to when you opened the screen: the card returns with exactly the schedule it had (kept in the answer's history row, `fsrs_before_json`), the answer disappears from the history, and a new card gets its slot in today's limit back. To make that possible, answers made during a visit are held back from upload until you leave the screen; if the app is closed first they are sent on the next launch. Home still counts them as waiting to sync. Once you have left, an answer is sent and can no longer be undone (practice mode, which records nothing, has an undo too, for the queue). Pausing a card is undone with Resume in the deck.
- **Study ahead:** when you are all caught up, offer the cards due in the next 1, 3 or 7 days. They are answered and scheduled normally (FSRS already gives an early review a smaller boost for a card you still remember well).
- **Practice:** the **All N** button on a deck or folder goes through every card that is not paused, due or not. It records nothing and changes no schedule; a missed card goes to the back of the queue.
- **Pausing and problem cards:** a card missed 8 or more times (Anki's default "leech" limit) gets a notice on the study screen. **Pause this card** (study screen, or the deck's card list, where it is also resumed) stops it being offered, counted or practised; it keeps its schedule and history. Paused cards are remembered on **this phone only** (a table the sync never touches, so a re-download keeps them); a reinstall or another phone forgets them.
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
   - Physical phone: `http://<your computer's Wi-Fi address>:8000` (start the server with `--host 0.0.0.0`; allow it through the Windows firewall). The app sends your token with every request, so it refuses to use plain `http://` for an address on the internet; `http://` is accepted only for your own machine or network (localhost, `10.x`, `172.16-31.x`, `192.168.x`, `100.64-127.x`, `.local` and bare computer names), and an installed app needs `https://`. An address typed without a scheme gets `https://` unless it is one of those.
   - Android emulator: `http://10.0.2.2:8000`.
   - Web preview: `http://localhost:8000`, and start the backend with `SNAPNOTES_CORS_ORIGINS=http://localhost:8081`.

Plain `http://` works in Expo Go and the web preview. Installed builds (Milestone 4) will need the backend behind HTTPS.

## Navigation and Stats

A bar at the bottom moves between **Decks** (Home), **Stats** and **Settings**. Everything else (a deck, studying, adding notes, reviewing) opens above the bar and covers it; going back returns to it. Until the server is set up the bar is hidden and only Settings shows. The three tabs live in `src/app/(tabs)/`.

**Stats** is worked out on the phone from the answers it has saved, so it works offline and needs no server. Pick the last 7, 30 or 90 days:

- **Retention:** of the answers given to cards you already knew (in review), the share that were not "Missed it". Practice mode and a card's first answers are not counted. Under 20 such answers the page says the number is too small to rely on.
- **Cards learned** (studied at least once, of all cards, and how many are mature, waiting 21 days or more), **days studied** with the current and longest **streak**, **answers**, and **time studied** (each answer counts at most a minute, so a phone left open does not add time).
- **Graphs:** answers per day, retention week by week (last 8 weeks), cards by stage (new, learning, young, mature), cards due in the next 14 days (overdue counted under Now; paused and new cards left out) and a 12-week study calendar. They are drawn with plain views, so there is no chart library.
- A day starts at 4 a.m., as everywhere else in the app. "Days studied" counts days with at least one answer, not app opens.
- **Limit:** the stats come from this phone's saved answers. The server keeps your reviews too but does not send them back, so a reinstalled app or a new phone starts with empty stats, and switching servers clears them.

## Managing decks

Each deck on the home screen is one line: a **☰** button, its name over a short summary (cards, what is due and new, anything needing a look), and a **Study** button when something is due. Tap the name to open the deck; **☰** opens the deck's settings. The same sheet is also reachable from inside a deck, with **Settings** at the top right. From it you can:

- **Rename** the deck.
- **Add notes to this deck.** Opens the upload screen aimed at the deck ("Add to “…”"). The new cards arrive as drafts in that deck for you to review, the deck's name stays, and any card that repeats one already in the deck is set aside with "Already in this deck." (only near-identical questions are caught, not reworded ones). The job screen says "Adding to “…”" and counts only this upload's cards.
- **Type a card.** One question and answer you write yourself, either "Question and answer" or "Fill in the blank" (wrap the answer like `{{c1::answer}}`; an explanation is optional). It is approved straight away, labelled "Added by you", and can be studied after the sync that starts when you save.
- **Merge another deck into this one.** Pick a deck from the list; after an "are you sure" that says how many cards move, all its cards (including ones still waiting for review) join this deck, your study progress on them is kept, and the other deck is removed. This deck keeps its title. It cannot be undone.
- **Practice all N cards** (from Home's menu), and **Delete deck**, after an "are you sure" that says how many cards go with it (this needs a connection, because the server holds the library). The delete button used to sit on every deck's line; it is in the menu so it is not hit by mistake.

**Reordering:** with two or more decks, press and hold anywhere on a deck's line for about half a second (a phone with a vibration motor gives a small buzz), then drag up or down. The other decks slide out of the way and the deck lands where you let go (a drag shorter than half a card changes nothing). A quick tap still opens the deck, buttons on the card still work, and scrolling the page is unaffected until a hold is up. Screen readers get "Move up" and "Move down" actions on the deck's title. The order is saved on this phone only (it is a preference, not part of the library, so it is not synced); a deck you have not placed yet, such as a new one, appears at the top. The page does not scroll while you hold a deck, so with a very long list, drag in steps.

All of these need a connection, like rename and delete, because the server holds the library; studying stays offline.

## Multiple choice

Turn a flashcard into a multiple choice question with the **Multiple choice** switch at the top of the study screen (remembered on this phone). With it on, each question shows 4 or 5 options under the card (set in **Settings → Studying**, default 4):

- **Pick one:** the right answer turns green, a wrong pick turns red with the right one shown, and the usual four rating buttons appear with the sensible one **suggested** (Got it after a right pick, Missed it after a wrong one). You still rate yourself, so a lucky guess does not count as knowing it. **I don't know** (or tapping the card) shows the answer with no suggestion, and swiping works as before.
- **Where the wrong options come from:** the card's own wrong answers first, then, if it has too few, answers borrowed from the **other cards in the same deck** (about as long as the right one, never a copy of it; switch this off in Settings). A card that still cannot make a sensible question (for example the deck has only one card) is studied the usual way, and so is every fill-in-the-blank card.
- **Writing them yourself:** a **Wrong answers** section in the card editor and on the Add a card screen, with up to 4 fields. The right answer is the card's back, and a wrong answer that matches it is refused.
- **Having the AI suggest them:** **Suggest with AI** in the card editor (one card, fills the fields for you to edit), or **Multiple choice answers** on a deck's page: it suggests for the cards that lack them, shows each suggestion editable with a switch to leave it out, and saves only what you keep. The free AI can write weak wrong answers, which is why you always review them first; typing them or borrowing from other cards does not need the AI.

All the multiple choice settings live on this phone only; the wrong answers themselves are part of the card, so they sync like the rest of the library. Suggesting needs a connection; studying does not.

## Finding a deck among many

Once there are six or more decks and folders, Home grows two tools under "Your decks":

- **Search:** type part of a deck's name or topic, or a folder's name. A folder that matches shows all its decks; a folder holding a matching deck shows just those decks; folders are open while you search. Press-and-hold reordering is switched off while a search is active, because the list is then a filtered view.
- **All decks:** opens every folder and deck in one searchable list (folders open, loose decks under "Not in a folder"). Tap a deck to open it, or its **Study** button to start. It works from what the phone holds, so it works offline.
- **Expand all / Collapse all** (with two or more folders) opens or closes every folder at once.

## Folders

Folders group decks on the same subject, such as **Biology** holding Bio I and Bio II, or **Programming** holding Python, SQL and Git. They are kept on the server and synced like decks, so a new phone sees them too.

- **Make one:** **New folder** beside "Your decks". A folder's ☰ renames it or deletes it; deleting a folder never deletes its decks (they simply stop being in a folder, and the question says so).
- **Put a deck in one:** the deck's ☰ → **Folder** → pick a folder (or make a new one right there). This always works. As a shortcut, **press and hold a deck and drop it on a folder's card**; the folder's border lights up when letting go will file it. Drop a deck on the **Not in a folder** heading to take it out. A drag that lands anywhere else only reorders the deck among the decks of its own list.
- **What Home shows:** folders first, then (once any folder exists) the decks that are in none. Folders start **closed**, so with many of them Home is a short list of names; tap a folder to open or close it (remembered on this phone), or use **Expand all / Collapse all**. A folder's line shows how many decks and cards it holds and what is due and new across all of them; the "new" figure respects your daily new-card limit.
- **Study a whole folder:** the **Study** button on its line studies every deck in it together; **Practice all N cards** is in its ☰ menu.
- Folders are listed A to Z. The order of decks inside a folder is the order you dragged them into on this phone. Making, renaming, deleting and filing need a connection (like rename and merge); studying stays offline.

## Appearance

**Settings → Appearance** chooses **Same as phone** (the default: light or dark follows the phone's own setting), **Light** or **Dark**. The choice applies to the whole app straight away and is remembered on this phone. The dark palette is the `dark` set in `src/constants/theme.ts`; the choice itself is in `src/lib/appearance.ts` and `appearance-provider.tsx`. The colours the system draws itself (alert dialogs, the keyboard) follow the choice too.

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
