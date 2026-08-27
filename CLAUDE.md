# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

SeatRacer is a rowing lineup analysis tool: seat racing results in, each
athlete out as a linear factor on boat pace with uncertainty, used to compare
athletes within a side and predict future lineup speeds. It is a browser-only
static app (React + Vite + TypeScript); all computation runs in Web Workers.
There is no server and no backend.

Style and engineering conventions follow the house guide at
`C:\projects\WEB_APP_STYLE_GUIDE.md`. Notable house rules: no icons or emojis
anywhere, pill groups for small choice sets, dense sortable tables,
token-driven dark mode, plain-words metric explanations.

## Commands

```bash
npm run dev         # dev server at http://localhost:8088
npm test            # engine test suite (golden fixtures + property tests)
npm run typecheck   # strict TypeScript check
npm run build       # static production build in dist/
```

## Architecture

- `src/engine/` is pure TypeScript with no DOM access: parsing, prep,
  weighting, design matrix, solvers (SVD minimum-norm WLS, Huber IRLS, Lp
  IRLS, ridge-toward-center), walk-forward evaluation, influence, and derived
  display stats. Everything user-visible flows from `fitModel` (model.ts) or
  `walkForward` (evaluate.ts).
- `src/workers/` wraps the engine for off-main-thread use; `src/ui/` is the
  React app with one global stylesheet (`src/index.css`).
- Bundled datasets live in `public/data/` (CSV schema: Race Session (date),
  Piece, KM, Rigging, Personnel, Result).

## Statistical invariants (do not break)

- Port and starboard athletes are separate parameters; never compare or rank
  across sides anywhere in the UI.
- Coefficients and shell effects are displayed only as gaps behind the leader
  (plus an uncertainty half-width), never as absolute numbers.
- The design matrix is rank-deficient by construction; solvers use a
  minimum-norm solution with a deliberate 1e-8 relative singular-value cutoff
  (see solve.ts for why statsmodels' 1e-15 is inside the noise band).
- Model evaluation is forward-in-time only, scored on within-piece margins
  (piece conditions cancel); same-day and future-day horizons are reported
  separately.
- Coxswains are excluded from the model by default.

## The golden-fixture gate

`fixtures/*.json` are committed outputs of the original Python engine
(pandas + statsmodels, deleted from the working tree; see git history at tag
points before the rewrite). `src/engine/golden.test.ts` asserts the TypeScript
engine reproduces them. Never regenerate or edit fixtures to make a test pass;
a mismatch means the engine changed behavior. The engine's prep formulas
(weights, clips, fractions) are 1:1 ports and are fixture-verified.
