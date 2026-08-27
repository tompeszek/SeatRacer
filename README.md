# SeatRacer

Rowing lineup and seat-racing analysis, as a browser-only static app. Race
results go in; each athlete comes out as a linear factor on boat pace, with
uncertainty, used to compare athletes within a side and predict future lineups.
Everything runs in the browser (Web Workers); there is no server.

## Run locally

```bash
npm install
npm run dev        # http://localhost:8088
```

`npm run build` produces the static site in `dist/`; `npm test` runs the engine
test suite; `npm run typecheck` checks types.

## Architecture

```
index.html               theme-before-paint bootstrap
src/
  engine/                pure TypeScript, no DOM; fully unit-tested
    parse.ts             CSV -> race rows
    prep.ts              times, shell classes, rigging suffixes, sorting
    weights.ts           recency / close-race / stern-bias weighting
    design.ts            design matrix (athlete fractions, shell + piece dummies)
    solve.ts             SVD minimum-norm WLS with t-based intervals
    robust.ts            Huber (statsmodels-RLM-equivalent) and Lp IRLS
    ridge.ts             ridge toward an arbitrary center (zero or erg-implied)
    model.ts             one fit() over the loss x shrinkage grid
    evaluate.ts          walk-forward evaluation (Model Lab)
    influence.ts         leave-one-out influence, over-time trends
    derived.ts           display stats: gaps, bias, pairs, correlations
  workers/               fit + evaluation Web Workers
  ui/                    React app, one global stylesheet (src/index.css)
public/data/             bundled example datasets
fixtures/                golden outputs from the original Python engine
```

## The model

The design is the standard lineup-attribution regression: boat pace per 500m
regressed on athlete indicator fractions plus shell class and piece dummies.
Port and starboard are separate parameters and are never compared to each
other; every display shows gaps behind the leader of a side, never absolute
coefficients.

User-selectable options, all combinations valid:

- **Error scoring:** squared (OLS, the default), Huber, or Lp with a chosen
  exponent (p below 1 discounts outlier pieces hard).
- **Shrinkage:** off, ridge toward zero, or ridge toward erg-implied paces
  (2k scores entered on the Athletes tab act as tiebreakers and priors).
- **Weighting:** recency halflife, close-race margins, stern bias, coxswain
  inclusion, shell class filter (identical formulas to the original engine).

The **Model Lab** answers "which options predict best" honestly: every model is
fit only on strictly earlier pieces and scored on predicted within-piece
margins, reported separately for same-day and future-day horizons.

## Provenance and verification

This app replaces a Python engine (pandas + statsmodels, NiceGUI frontend; see
git history). Before that engine was deleted, its outputs on all bundled
datasets under a matrix of settings were captured in `fixtures/`; the
TypeScript engine's test suite (`npm test`) reproduces them: OLS/WLS
coefficients, standard errors, and intervals; Huber RLM coefficients and H1
covariance; GLM coefficients. Ridge and Lp, which have no Python counterpart,
are property-tested against known limits (lambda 0 = OLS, p = 2 = OLS,
p = 1 = median).

## Deploy

The build is static files; any static host works. `npm run build`, then serve
`dist/`.
