# Phase 6 results — UX pass (theme, navigation, profiles, settings, help)

Date: 2026-09-15. Device: RedMagic 10 Pro (NX789J), Android 15, 16 GB, USB-connected.
Scope: turn the dev-harness UI (7 top tabs, default MaterialTheme) into a proper app:
home dashboard + bottom navigation, full theming, learner profile, memory spaces,
performance presets, settings, in-app documentation. All verified on-device via adb.

## 1. What shipped

- **Theme** (`ui/theme/`): light/dark/system + accent presets Dynamic (Material You),
  Indigo, Sakura, Matcha. Live-applied from Settings → Appearance. Fixed during bring-up:
  per-instance `AppSettings.changes` flows meant root-theme writes from deep screens didn't
  propagate — now driven by a `SharedPreferences.OnSharedPreferenceChangeListener`, so any
  instance's write recomposes the root (verified: Sakura applied app-wide on-device).
- **Navigation** (`MainActivity.kt` rewrite, `ui/Routes.kt`, navigation-compose 2.9.2):
  bottom NavigationBar with Home / Chat / Tutor / Memory / Settings; Practice, Brain, Graph,
  Diagnostics (spikes), Help, and settings subscreens are NavHost subroutes with back stack.
  ViewModels are scoped to the **activity** (not NavBackStackEntry) so chat/tutor/sim state
  survives navigation. `SpikeScreen` moved to `spikes/ui/`.
- **Home** (`home/`): greeting (learner name), active space, date; quick actions; status card
  (selected/resident model, mode, backend, preset, both store ids); memory stats card
  (documents, chunks, entities, edges, Hebbian concepts/edges, fading queue); ARM-E card when
  the tutor engine has history.
- **Settings** (`settings/`): Appearance, Performance (presets write the existing `rag` prefs;
  overrides mark "custom"), Learner profile, Memory spaces, Model & memory (resident model,
  unload, installed GGUFs with sizes), Data (clear Hebbian memory / documents+graph / practice
  state, all double-confirmed and space-scoped), Diagnostics, Help, About.
- **Learner profile → tutor**: `learnerPromptBlock()` appended to the persona system prompt at
  model load (`HebbianEngineFactory.ensureModelsLoaded`); saving the profile unloads the tutor
  LLM so the next turn picks it up.
- **Memory spaces** (`profile/SpaceManager.kt`): per-space RAG db, entity graph (Ladybug +
  SQLite fallback), Hebbian graph, and practice state under `files/spaces/<id>/`; models,
  theme, learner profile, eval corpus stay global. Switch = drop engines (`HebbianEngineHolder
  .reset()`), unload LLM, close both store factories, set active, recreate the activity.
- **Help** (`help/HelpScreen.kt` + `assets/help.md`): in-app docs with a minimal markdown
  renderer (headings/bullets/bold/code/dividers) — personas, ARM-E glossary, spaces, models
  (incl. adb push), diagnostics, troubleshooting, privacy.
- **`README.md`** (new — repo had none).

## 2. Space migration (legacy → spaces/personal)

One-time, idempotent, lock-guarded; runs inside `SpaceManager.activeDir()` so it always
precedes any store open. **Bug found on-device**: the Ladybug stores' WAL files
(`graph/ladybug.wal`, `hebbian/hebbian.lbug.wal`) weren't moved with their main files, so the
entity graph opened empty (data stranded in the unmoved WAL) and the Hebbian Ladybug store
failed replay ("Corrupted wal file") and fell back to SQLite. Fixed in the migration and
repaired on-device (WALs moved into place): 97 entities / 503 edges recovered. The Hebbian
store was deliberately reset to a fresh LadybugDB instance (its pre-migration content was
smoke-test data; both Hebbian files wiped, `chosen=ladybugdb-hebbian` on next launch).

## 3. On-device verification

- Home renders with real counts in both spaces; Test space created → switched (zeros, both
  stores ladybugdb) → switched back (Personal counts intact). Space isolation works.
- Appearance: Sakura accent applied live app-wide (screenshot-verified).
- `hebbian_ingest` spike: **PASS 6/6 on ladybugdb-hebbian** (Test space).
- `hybrid_eval`: **PASS** vs baseline 20260915-005156 — vector recall@5 1.00, hybrid 1.00,
  gate 0.75 both; saved history/20260915-042704.json. No regression from the rework.
- Help page renders all markup correctly.

## 4. Notes / follow-ups

- Space directories are created lazily on first data access (not at "Add space" time).
- The settings-hub Deep screens use per-instance `AppSettings` for writes; reads are always
  backed by the same prefs file, and recomposition is listener-driven (see §1).
- Tutor↔Chat↔Brain model reloads on tab switch remain (per-session system prompts, by design).
- Spikes operate on the **active** space; the Hebbian spikes still wipe its Hebbian memory —
  run them on a scratch space.
