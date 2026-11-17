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
