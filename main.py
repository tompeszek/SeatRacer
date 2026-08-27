"""SeatRacer entrypoint (NiceGUI).

Run locally:      python main.py
On Railway:       reads $PORT and binds 0.0.0.0 (see railway.toml / Procfile)

The Individual tab parallelises its refits with a ProcessPoolExecutor. Under the
spawn start method, each worker re-imports this module as ``__mp_main__`` – so the
server launch is guarded by ``if __name__ == "__main__"`` (and the heavy app import
lives inside ``main()``) to make sure workers never start their own server.
"""
import os
import sys


def main():
    # Athlete names carry superscript rigging marks (ᵖ ˢ ᶜ ˣ); make sure console
    # logging can encode them on Windows terminals (cp1252 otherwise raises).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    from nicegui import ui
    import seatracer.ng.app  # noqa: F401  -- registers the @ui.page("/") route

    ui.run(
        title="SeatRacer",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8088)),
        reload=False,
        show=False,
        storage_secret=os.environ.get("STORAGE_SECRET", "seatracer-local-dev-secret"),
    )


if __name__ == "__main__":
    main()
