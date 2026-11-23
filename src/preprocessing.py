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


# --------------------------------------------------------------------------
# 2. Rolling team/venue state (the only place features are computed, so
#    training and live prediction use identical logic)
# --------------------------------------------------------------------------
class _State:
    def __init__(self, window=10, elo_start=1500.0, k=24.0, prior=3.0):
        self.window, self.elo_start, self.k, self.prior = window, elo_start, k, prior
        self.elo = defaultdict(lambda: elo_start)
        self.rr_for = defaultdict(lambda: deque(maxlen=window))
        self.rr_against = defaultdict(lambda: deque(maxlen=window))
        self.results = defaultdict(lambda: deque(maxlen=window))
        self.played = defaultdict(int)
        self.v_n = defaultdict(int)
        self.v_sum = defaultdict(float)
        self.v_win = defaultdict(float)
        self.g_n, self.g_sum, self.g_win = 0, 0.0, 0.0

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if len(values) else np.nan

    def features(self, bat, bowl, venue, toss_winner, toss_decision, year):
        if self.g_n:
            g_avg, g_win = self.g_sum / self.g_n, self.g_win / self.g_n
            n = self.v_n[venue]
            v_avg = (self.v_sum[venue] + self.prior * g_avg) / (n + self.prior)
            v_win = (self.v_win[venue] + self.prior * g_win) / (n + self.prior)
        else:
            v_avg = v_win = np.nan
        return {
            "elo_bat": self.elo[bat], "elo_bowl": self.elo[bowl],
            "elo_diff": self.elo[bat] - self.elo[bowl],
            "bat_form_for": self._mean(self.rr_for[bat]),
            "bat_form_against": self._mean(self.rr_against[bat]),
            "bowl_form_for": self._mean(self.rr_for[bowl]),
            "bowl_form_against": self._mean(self.rr_against[bowl]),
            "bat_win_rate": self._mean(self.results[bat]),
            "bowl_win_rate": self._mean(self.results[bowl]),
            "bat_matches": self.played[bat], "bowl_matches": self.played[bowl],
            "venue_avg_first": v_avg, "venue_bat_first_win": v_win,
            "toss_won_by_bat_team": int(toss_winner == bat),
            "toss_chose_bat": int(toss_decision == "bat"),
            "year": year,
        }

    def update(self, m):
        bat, bowl, venue = m["bat_team"], m["bowl_team"], m["venue"]
        for runs, balls, batting, bowling in (
            (m["first_innings_runs"], m["first_innings_balls"], bat, bowl),
            (m["second_innings_runs"], m["second_innings_balls"], bowl, bat),
        ):
            if balls > 0:
                rate = runs / (balls / 6.0)       # runs per over, robust to short innings
                self.rr_for[batting].append(rate)
                self.rr_against[bowling].append(rate)

        won = m["bat_first_won"]
        won = 0.5 if pd.isna(won) else float(won)  # tie / no result
        expected = 1.0 / (1.0 + 10 ** ((self.elo[bowl] - self.elo[bat]) / 400.0))
        delta = self.k * (won - expected)
        self.elo[bat] += delta
        self.elo[bowl] -= delta
        self.results[bat].append(won)
        self.results[bowl].append(1.0 - won)
        self.played[bat] += 1
        self.played[bowl] += 1

        self.v_n[venue] += 1
        self.v_sum[venue] += m["first_innings_runs"]
        self.v_win[venue] += won
        self.g_n += 1
        self.g_sum += m["first_innings_runs"]
        self.g_win += won


# --------------------------------------------------------------------------
# 3. Public Preprocessor
# --------------------------------------------------------------------------
class Preprocessor:
    """Cleans the match table and builds pre-match features without leakage."""

    def __init__(self, matches):
        self.df = matches.copy()
        self._state = None

    def clean(self):
        required = ["venue", "bat_team", "bowl_team", "toss_winner", "toss_decision"]
        self.df = self.df.dropna(subset=required).copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values(["date", "match_id"]).reset_index(drop=True)
        return self.df

    def build_features(self):
        """Each row's features use only matches played BEFORE it."""
        state = _State()
        rows = []
        for m in self.df.to_dict("records"):
            rows.append(state.features(m["bat_team"], m["bowl_team"], m["venue"],
                                       m["toss_winner"], m["toss_decision"], m["date"].year))
            state.update(m)
        self._state = state
        self.df = pd.concat([self.df, pd.DataFrame(rows)], axis=1)
        return self.df

    def fixture_features(self, bat, bowl, venue, toss_winner, toss_decision, year=None):
        """Features for an upcoming match, from the state after the latest played match."""
        if self._state is None:
            raise RuntimeError("Call build_features() first.")
        year = year or int(self.df["date"].max().year)
        return pd.DataFrame([self._state.features(bat, bowl, venue, toss_winner, toss_decision, year)])[FEATURES]

    def training_frame(self, min_history=5):
        """Rows where both teams already have some history."""
        d = self.df
        return d[(d["bat_matches"] >= min_history) & (d["bowl_matches"] >= min_history)]

    @staticmethod
    def split_by_date(df, test_frac=0.2):
        """Chronological split: train on the past, test on the most recent matches."""
        df = df.sort_values("date")
        cut = int(len(df) * (1 - test_frac))
        return df.iloc[:cut], df.iloc[cut:]

    def known_teams(self):
        return sorted(self._state.played)

    def known_venues(self):
        return sorted(self._state.v_n)

    def elo_table(self):
        return pd.Series(dict(self._state.elo)).sort_values(ascending=False)
