# SeatRacer Rewrite Plan: Browser-Only TypeScript App

**Status: completed 2026-08-27.** All milestones done; the Python engine and
NiceGUI app are deleted (recoverable from git history), the golden fixtures
remain committed, and the gate passes. Changes agreed during the build:
coxswains excluded by default; port and starboard rendered as side-by-side
per-side tables; all effects displayed as gaps behind the leader rather than
absolute coefficients; the old Lineup Testing tab folded into New Lineup;
Optimal Lineups (a placeholder) and the Debug tab dropped as planned.

Agreed direction (2026-08-27): replace the NiceGUI/Python app with a static,
browser-only TypeScript app. No server, no roundtrips. All computation runs in
the browser in a background thread. The Python engine is deleted at the end,
after the TypeScript engine is verified against it numerically.

This app follows the house guide at `C:\projects\WEB_APP_STYLE_GUIDE.md` for
look, feel, writing, and engineering approach. Points where that guide shapes
this plan are called out inline; where the guide assumes a backend (realtime,
roles, offline warmers) the corresponding sections do not apply, since this app
has no server at all.

## 1. Goals

1. **Zero server.** The app is static files, deployable to any static host
   (GitHub Pages, Netlify). Railway and all deploy scripts go away.
2. **Defensible statistics.** The default model is plain OLS. Every option
   beyond that is an explicit, named, user-visible choice. No home-grown
   inference procedures: linear algebra and probability distributions come from
   vetted libraries, and the few closed-form formulas written by hand (the
   weighted least squares covariance) are verified against statsmodels outputs.
3. **Empirical model selection.** A built-in evaluation answers "which model
   settings predict best?" using forward-only (walk-forward) testing. This is
   the guide's Section 24 rule ("score it forward in time") built into the
   product.
4. **Numerical parity before deletion.** The Python engine's outputs on all
   bundled datasets become committed test fixtures. The TypeScript engine must
   reproduce them before any Python code is removed. This is the guide's
   Section 26 equivalence gate for ported logic.

## 2. Stack

- **Build:** Vite + React + TypeScript (strict). Plain React state; no state
  library unless needed.
- **Styling:** one global `index.css` driven by CSS custom properties, per the
  house guide, with the rowfest app as the look-and-feel reference. No
  component library. Roboto (400/500/700) bundled locally; **no icon font and
  no icons anywhere** (guide Section 2.3: icons are opt-in only, and none are
  requested for this app). Token-driven dark mode (`data-theme="dark"` on
  `<html>`, set before first paint) from the start, with the toggle in the
  header menu. Accent palette: the guide's reference crimson tokens; swapping
  the palette later is a one-place change.
- **Tables:** plain HTML tables with the guide's classes and density rules
  (`.table-card` horizontal-scroll wrapper, `.num` columns with tabular
  numerals, every column sortable, group separators). No table library.
- **Charts:** ECharts (npm package, bundled). It is a rendering library, not a
  component kit, and it is what the current app's charts already use.
- **Math:** `ml-matrix` for linear algebra (QR solve, matrix products);
  `@stdlib/stats` for t-distribution quantiles.
- **Compute:** Web Workers for model fitting and evaluation so the page never
  blocks. Data is small (hundreds of rows), so full evaluation grids run in
  well under a second.
- **Tests:** Vitest. Golden-fixture tests (section 6) plus unit tests for
  parsing and weighting helpers. Build and typecheck gate every commit.

## 3. Repository layout

```
src/
  engine/            pure TypeScript, no DOM access, fully unit-testable
    parse.ts         CSV -> Race rows  (schema: Race Session (date), Piece,
                     KM, Rigging, Personnel, Result)
    prep.ts          time parsing (mm:ss -> s), pace per 500m, shell class
                     detection, rigging suffixes on names (p/s/c/x marks)
    weights.ts       recency (halflife), close-race margin weights
    design.ts        design matrix: piece dummies, athlete columns as boat
                     fractions (1/n, or stern-weighted), shell class dummies
    solve.ts         weighted least squares (QR); ridge solve with an
                     arbitrary center vector (zero or erg-implied)
    robust.ts        Huber and Lp(p) fitting via iteratively reweighted
                     least squares (IRLS)
    inference.ts     coefficient covariance, standard errors, confidence
                     intervals
    model.ts         ModelSpec type + fit(): one entrypoint for every
                     loss x shrinkage combination
    evaluate.ts      walk-forward evaluation (section 5)
    predict.ts       lineup prediction and comparison
    derived.ts       per-athlete stats (rank, speed-behind, correlations),
                     athlete pairs (synergy), leave-one-out influence,
                     fitted-vs-actual, race balance
  workers/           fit worker, evaluation worker
  ui/                React app: header, sidebar controls, tabs
  index.css          global stylesheet: tokens, components, dark block
public/data/         bundled example datasets (the 6 CSVs)
fixtures/            golden outputs generated from the Python engine
tools/make_fixtures.py   fixture generator (deleted with the Python engine;
                         fixtures stay committed)
```

## 4. The model space

Two independent axes, selectable in the sidebar. Every combination is valid.

**Axis 1, error scoring (loss):**
| Option | Meaning | Fitting | Intervals |
|---|---|---|---|
| Squared (default) | classic least squares | one QR solve | classical t-based (exact statsmodels parity) |
| Huber(c) | squared for small misses, linear for large | IRLS | standard robust-regression covariance, validated against statsmodels RLM |
| Lp(p), p in (0, 2] | user-chosen exponent; p below 1 discounts outliers hard | IRLS, deterministically initialized from the squared-loss fit, fixed tolerance and iteration cap | bootstrap over pieces (resample pieces, refit, take percentile intervals) |

**Axis 2, shrinkage:**
| Option | Meaning |
|---|---|
| Off (default) | plain regression |
| Ridge toward zero, strength lambda | coefficients pulled toward 0; handles athletes whose data cannot separate them |
| Ridge toward erg scores, strength lambda | coefficients pulled toward each athlete's erg-implied pace; erg data acts as tiebreaker and fallback |

Default configuration = Squared + Off = **OLS**, the defensible baseline.

**Carried over unchanged** (as observation weights and design choices,
independent of both axes): recency halflife, close-race margin weighting,
stern-position athlete weighting, coxswain inclusion toggle, shell-class
filter. Formulas are ported 1:1 from `analysis_base.py`, including the clip
values (recency floor 0.1 and cap 10, closeness floor 0.1, margin cap 12).

**Dropped:** the custom gradient-descent model (its two purposes, outlier
resistance and erg-score anchoring, are now explicit options); the
correlation-threshold athlete dropping as a *model* mechanism (correlation
info remains displayed as a diagnostic; ridge handles the underlying problem);
and all hidden models (RLM/WLS/ElasticNet/RF/XGBoost/TrueSkill).

Determinism (guide Section 24): fitting and evaluation use no randomness
except the Lp bootstrap, which uses a fixed seeded generator so reruns are
comparable.

## 5. Walk-forward evaluation ("Model Lab" tab)

The centerpiece new feature. For a chosen set of candidate model configurations:

1. Order all pieces chronologically (session date, then piece number).
2. For each piece k past a warm-up minimum: fit each candidate on all strictly
   earlier pieces, predict the boats in piece k.
3. Score **within-piece margins**: predicted vs actual pace differences between
   boats in the same piece. This cancels the piece's unknown conditions
   (weather, current), which no model can know in advance.
4. Report results per candidate, **split by horizon**:
   - *Same Day*: predicting a later piece of a day already partly seen
   - *Future Day*: predicting the first pieces of a new day
   A model may win one horizon and not the other; both are shown.
5. New-athlete handling: if a boat in piece k contains an athlete never seen
   before, erg-shrinkage models predict them at their erg-implied value; other
   models exclude that boat pair from scoring. Excluded counts are reported so
   the comparison is transparent.
6. Output: a leaderboard table plus a drill-down of the worst-predicted pieces
   per model.

Metrics follow the guide's plain-words rule (Sections 2.5 and 24): the
leaderboard reports the median absolute margin miss in seconds and the
percentage of boat pairs where the model named the faster crew, each defined
in one sentence of prose beside the table. No bare "RMSE 11.2" cells, and no
stat-tile grids; results are a sortable table.

Runs in a worker with a progress indicator. Candidate set = a small default
grid (OLS; Huber; Lp p=0.5; each with and without erg-ridge at a few lambda
values) plus whatever configuration the user currently has selected.

## 6. Verification (before any Python is deleted)

1. `tools/make_fixtures.py` runs the **existing Python engine** on all 6
   bundled datasets under a matrix of settings (weights on/off, coxswains
   on/off, stern bias levels) and writes JSON fixtures: prepped rows, design
   matrix, coefficients, standard errors, confidence intervals, for OLS/WLS
   and for Huber (statsmodels RLM).
2. Vitest golden tests assert the TypeScript engine reproduces: coefficients
   within 1e-8, standard errors and CI bounds within 1e-6.
3. Helper-level fixtures for time parsing, shell class detection, rigging
   suffix assignment, and each weighting formula.
4. Ridge and Lp have no Python counterpart; they get property tests instead
   (lambda=0 reduces to OLS; p=2 matches OLS; p=1 matches median regression on
   a hand-checkable small case; the ridge solution moves monotonically toward
   the center as lambda grows).

## 7. UI: tabs carried over

All user-visible text follows the guide's writing rules: American spelling,
Title Case labels, full words, no emojis, no em dashes, no middot runs, no
invented instructional copy, explanations in prose beside tables rather than
inside cells.

Phase A (core): **Data** (bundled datasets, CSV upload, editable grid),
**Performance** (coefficients, CIs, ranks, speed-behind: the main results
view, as a dense sortable table, not tiles), **New Lineup** (prediction), and
the model controls. Controls follow the rowfest pattern: an `.options-box` of
labeled option rows, each a fixed-width label plus a segmented pill group
(loss, shrinkage, recency, close races, stern bias, coxswains); a native
select only for genuinely long lists. No dropdowns where pills fit, and no
icons anywhere.

Phase B: **Model Lab** (section 5).

Phase C: **Individual** (leave-one-out influence, now just synchronous loops
in the worker; the ProcessPoolExecutor machinery disappears), **Correlations**,
**Synergies** (pairs), **Fairness**, **Validation** (fitted vs actual, race
balance), **Over Time**, **Athletes** (erg score entry), **Lineup Testing**.

Dropped: Optimal Lineups (was already a placeholder) and the Debug tab.

Navigation: the guide's tab bar in its text-only form (small uppercase labels,
accent underline on active, scroll-with-cues on overflow). Mobile follows the
guide's 640px rules: densified tables, stacked controls, overlay-scroll
modals.

Persistence: uploaded datasets and settings in localStorage (guarded reads),
with export and import buttons as the reliable path. Dark mode preference in
localStorage per guide Section 21.

## 8. Milestones

Each milestone ends with tests green, typecheck clean, and the app runnable.

1. **Fixtures first.** Write `tools/make_fixtures.py`, generate and commit
   fixtures from the current Python engine.
2. **Engine core.** Scaffold the Vite project and the token stylesheet;
   implement parse, prep, weights, design, solve, inference; golden tests for
   OLS/WLS pass.
3. **Model options.** robust.ts (Huber golden-tested against RLM; Lp
   property-tested), ridge with center vector, ModelSpec and fit().
4. **Phase A UI.** Data, Performance, and New Lineup tabs against the worker,
   in both themes.
5. **Model Lab.** evaluate.ts plus UI, with horizon split, plain-words
   metrics, and exclusion counts.
6. **Phase C UI.** Remaining tabs.
7. **Cleanup.** Delete the `seatracer/` Python package, `main.py`, NiceGUI
   dependencies, Railway files, and `tools/make_fixtures.py` (fixtures
   remain). Update README and CLAUDE.md. Set up static deploy with auto-deploy
   on push.

Optional after milestone 7: PWA shell caching (guide Section 15) so the app
opens with no connectivity. With no backend, this is precache-only and cheap.
