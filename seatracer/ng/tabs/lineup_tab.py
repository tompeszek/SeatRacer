"""Lineup Testing tab: build several lineups by hand and compare predicted speed."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from seatracer.utils.helpers import rig_map_reverse
from ..ui_common import df_to_grid


def _seat_labels(rower_count: int):
    labels = []
    for t in range(rower_count):
        j = rower_count - 1 - t
        if j == rower_count - 1:
            labels.append("Stroke")
        elif j == 0:
            labels.append("Bow")
        else:
            labels.append(f"Seat #{j + 1}")
    return labels


def build(dash):
    s = dash.state
    analysis = s.analysis
    athletes = analysis.final_results["athletes"].index.tolist()
    rowers_list = sorted(a for a in athletes if not a.endswith("ᶜ"))
    coxes_list = sorted(a for a in athletes if a.endswith("ᶜ"))
    classes_list = sorted(analysis.final_results["shell_classes"].index.tolist(), reverse=True)

    if not rowers_list or not classes_list:
        ui.label("Not enough model data to test lineups.").classes("text-gray-500")
        return

    ui.label("Lineup testing").classes("text-base font-semibold")
    ui.label("Select rowers and boat class to test different lineups.") \
        .classes("text-xs text-gray-500 italic")

    with ui.row().classes("items-center gap-2"):
        ui.label("Number of crews").classes("text-sm")
        ui.toggle([2, 3, 4, 5, 6], value=s.lt_boat_count,
                  on_change=lambda e: _set_count(int(e.value))).props("no-caps dense")

    def _set_count(n):
        s.lt_boat_count = n
        dash.rebuild("Lineup Testing")

    # Keep the persisted lineup structures in sync with the requested count.
    while len(s.lt_lineups) < s.lt_boat_count:
        s.lt_lineups.append({"boat_class": classes_list[0], "seats": [], "cox": None})
    s.lt_lineups = s.lt_lineups[: s.lt_boat_count]

    def _normalize(lineup):
        if lineup["boat_class"] not in classes_list:
            lineup["boat_class"] = classes_list[0]
        bc = lineup["boat_class"]
        rc = int(bc[0])
        has_cox = bc.endswith("+") and len(coxes_list) > 0
        seats = [a if a in rowers_list else rowers_list[0] for a in lineup.get("seats", [])][:rc]
        while len(seats) < rc:
            seats.append(rowers_list[min(len(seats), len(rowers_list) - 1)])
        lineup["seats"] = seats
        if has_cox:
            if lineup.get("cox") not in coxes_list:
                lineup["cox"] = coxes_list[0]
        else:
            lineup["cox"] = None
        return rc, has_cox

    valid_lineups = []
    with ui.row().classes("w-full items-start gap-4"):
        for i, lineup in enumerate(s.lt_lineups):
            rc, has_cox = _normalize(lineup)
            with ui.column().classes("flex-1 min-w-[210px] max-w-[280px] gap-1 border rounded p-2"):
                ui.label(f"Lineup #{i + 1}").classes("font-semibold")

                def _on_class(e, idx=i):
                    s.lt_lineups[idx]["boat_class"] = e.value
                    dash.rebuild("Lineup Testing")
                ui.select(classes_list, value=lineup["boat_class"], label="Boat class",
                          on_change=_on_class).props("dense outlined").classes("w-full")

                if has_cox:
                    def _on_cox(e, idx=i):
                        s.lt_lineups[idx]["cox"] = e.value
                        dash.rebuild("Lineup Testing")
                    ui.select(coxes_list, value=lineup["cox"], label="Cox",
                              on_change=_on_cox).props("dense outlined").classes("w-full")

                for seat_idx, label in enumerate(_seat_labels(rc)):
                    def _on_seat(e, idx=i, sidx=seat_idx):
                        s.lt_lineups[idx]["seats"][sidx] = e.value
                        dash.rebuild("Lineup Testing")
                    ui.select(rowers_list, value=lineup["seats"][seat_idx], label=label,
                              on_change=_on_seat).props("dense outlined").classes("w-full")

                seats = lineup["seats"]
                if len(set(seats)) != len(seats):
                    ui.label("Duplicate rowers selected.").classes("text-xs text-amber-700")
                rig = {"p": 0, "s": 0, "c": 0, "x": 0}
                for r in seats:
                    rig[rig_map_reverse.get(r[-1], "x")] += 1
                if rig["p"] != rig["s"] and "x" not in lineup["boat_class"]:
                    ui.label("Port and starboard must be equal.").classes("text-xs text-amber-700")

                # Cox (if any) is appended; with even weighting its position doesn't
                # affect the predicted time, only its own coefficient contributes.
                personnel = list(seats) + ([lineup["cox"]] if (has_cox and lineup["cox"]) else [])
                try:
                    predicted = analysis.predict_lineup(personnel, lineup["boat_class"],
                                                        return_formatted=True)
                    ui.label(f"Predicted: {predicted} / 500m").classes("font-medium text-blue-800")
                    valid_lineups.append({"name": f"Lineup #{i + 1}", "personnel": personnel,
                                          "shell_class": lineup["boat_class"]})
                except Exception as exc:  # noqa: BLE001
                    ui.label(f"Prediction error: {exc}").classes("text-xs text-red-600")

    if len(valid_lineups) > 1:
        ui.separator().classes("my-3")
        ui.label("Lineup comparison").classes("text-base font-semibold")
        try:
            comparison_df, details = analysis.compare_lineups(valid_lineups)
            with ui.column().classes("w-full max-w-[900px]"):
                df_to_grid(comparison_df.reset_index(drop=True))

                with ui.expansion("Detailed breakdown of all lineups").classes("w-full mt-2"):
                    for detail in details:
                        ui.label(f"{detail['name']} - {detail['formatted_time']}") \
                            .classes("font-semibold mt-2")
                        ui.label(f"Shell ({detail['shell_class']}): "
                                 f"{round(detail['shell_contribution'], 1)} s").classes("text-sm")
                        ui.label(f"Athletes total: {round(detail['athlete_contribution'], 1)} s") \
                            .classes("text-sm")
                        rows = []
                        rc = len(detail["personnel"])
                        for ad in detail["athlete_details"]:
                            pos = ad["position"]
                            athlete = ad["athlete"]
                            if athlete.endswith("ᶜ"):
                                pos_name = "Cox"
                            else:
                                pos_name = ("Stroke" if pos == rc else "Bow" if pos == 1
                                            else f"Seat #{pos}")
                            rows.append({
                                "Position": pos_name, "Athlete": athlete,
                                "Weight": round(ad["weight"], 2),
                                "Coefficient": round(ad["coefficient"], 1),
                                "Contribution": round(ad["contribution"], 1),
                            })
                        df_to_grid(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001
            ui.label(f"Error comparing lineups: {exc}").classes("text-red-600")
