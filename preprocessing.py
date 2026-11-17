import json
import zipfile
from collections import defaultdict, deque
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from exceptions import DataUnavailableError

FEATURES = [
    "elo_bat", "elo_bowl", "elo_diff",
    "bat_form_for", "bat_form_against", "bowl_form_for", "bowl_form_against",
    "bat_win_rate", "bowl_win_rate", "bat_matches", "bowl_matches",
    "venue_avg_first", "venue_bat_first_win",
    "toss_won_by_bat_team", "toss_chose_bat", "year",
]
SCORE_TARGET = "first_innings_runs"
WINNER_TARGET = "bat_first_won"


# --------------------------------------------------------------------------
# 1. Parsing Cricsheet JSON -> one row per match
# --------------------------------------------------------------------------
def _innings_summary(innings):
    """Return (runs, wickets, legal_balls) for one innings."""
    runs = wickets = balls = 0
    for over in innings.get("overs", []):
        for d in over["deliveries"]:
            runs += d["runs"]["total"]
            wickets += len(d.get("wickets", []))
            extras = d.get("extras", {})
            if "wides" not in extras and "noballs" not in extras:
                balls += 1
    return runs, wickets, balls


def _parse_match(match_id, data, fmt):
    info = data.get("info", {})
    if info.get("gender") != "male" or info.get("team_type") != "international":
        return None
    outcome = info.get("outcome", {})
    if outcome.get("method"):          # rain-affected (D/L) matches distort totals
        return None
    innings = [i for i in data.get("innings", []) if not i.get("super_over")]
    if len(innings) != 2:
        return None

    r1, w1, b1 = _innings_summary(innings[0])
    r2, w2, b2 = _innings_summary(innings[1])
    bat, bowl = innings[0]["team"], innings[1]["team"]
    winner = outcome.get("winner")
    toss = info.get("toss", {})
    return {
        "match_id": match_id,
        "date": info["dates"][0],
        "format": fmt,
        "venue": info.get("venue"),
        "bat_team": bat,
        "bowl_team": bowl,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "first_innings_runs": r1, "first_innings_wickets": w1, "first_innings_balls": b1,
        "second_innings_runs": r2, "second_innings_wickets": w2, "second_innings_balls": b2,
        "winner": winner,
        "bat_first_won": (1.0 if winner == bat else 0.0) if winner else np.nan,
    }


def parse_matches_zip(zip_path, fmt):
    """Read every match JSON inside a Cricsheet zip into a DataFrame."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    data = json.load(f)
                row = _parse_match(PurePosixPath(name).stem, data, fmt)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if row:
                rows.append(row)
    if not rows:
        raise DataUnavailableError(f"No usable matches found in {zip_path}.")
    return pd.DataFrame(rows)
