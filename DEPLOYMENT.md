# SeatRacer Railway Deployment Guide

The app is a NiceGUI (FastAPI/uvicorn) web app. It runs as a single process and
deploys on Railway with no extra configuration.

## Deploy to Railway

1. **Connect the repository**
   - Railway → "New Project" → "Deploy from GitHub repo" → select this repo.
2. **Configuration is automatic**
   - `railway.toml` (nixpacks builder) + `Procfile` define the start command
     `python main.py`.
   - `requirements.txt` lists the Python dependencies.
   - `runtime.txt` pins the Python version.
   - `main.py` binds `0.0.0.0` and reads `$PORT`; Railway provides `$PORT`.
3. **Optional env vars**
   - `STORAGE_SECRET` – secret used to sign per-client storage (recommended in
     production). Defaults to a local dev value if unset.

## Using the deployed app

1. Open the deployed URL.
2. On the **Data** tab, load an example dataset or upload your own CSV
   (max ~100 MB; the in-browser uploader streams the file to the server).
3. The analysis tabs populate automatically once data is loaded. Adjust the model
   and weights in the sidebar — each change re-fits the model once and refreshes
   the visible tab.

## CSV format

The racing CSV needs these columns:

```
Race Session (date), Piece, KM, Rigging, Personnel, Result
```

- `Rigging` uses per-seat codes joined by `/` (e.g. `c/p/s/p/s/p/s/p/s` for an 8+).
- `Personnel` lists athlete names in the same order, joined by `/`.
- `Result` is the time as `MM:SS` or `MM:SS.s`.

See the example files in `seatracer/data/` for the exact format. Erg files
(`seatracer/erg_data/`) use `Athlete, 2k Erg` with erg times as `m:ss.s`.

## Local development

```bash
cd lit_seatracer
pip install -r requirements.txt
python main.py        # http://localhost:8088
```

For live-reload during development, you can run NiceGUI with reload enabled by
temporarily setting `reload=True` in `main.py`'s `ui.run(...)` call.

## Troubleshooting

- **Build fails** – ensure all imports are covered by `requirements.txt`.
- **App starts but is blank** – check Railway logs; the engine logs benign
  statsmodels `RuntimeWarning`/`PerfectSeparationWarning` lines for small datasets,
  which are expected and not errors.
- **Port issues** – the app must read `$PORT`; this is already handled in `main.py`.
