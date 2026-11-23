import os
import re
import requests
import pandas as pd
from exceptions import DataUnavailableError


class DataManager:
    """Downloads (once), caches and locates Cricsheet zips, processed tables and venue texts."""

    def __init__(self, cache_dir="data/raw", processed_dir="data/processed"):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.urls = {
            "t20": "https://cricsheet.org/downloads/t20s_json.zip",
            "odi": "https://cricsheet.org/downloads/odis_json.zip",
        }

    def zip_path(self, fmt):
        return os.path.join(self.cache_dir, f"{fmt}s_json.zip")

    def processed_path(self, fmt):
        return os.path.join(self.processed_dir, f"{fmt}_matches.csv")

    def fetch_zip(self, fmt, refresh=False):
        """Use the cached zip unless refresh=True; fall back to the cache if the download fails."""
        if fmt not in self.urls:
            raise ValueError(f"Unknown format '{fmt}'. Choose from {list(self.urls)}.")
        path = self.zip_path(fmt)
        if os.path.exists(path) and not refresh:
            return path
        tmp = path + ".part"
        try:
            with requests.get(self.urls[fmt], stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(tmp, path)
            return path
        except (requests.RequestException, OSError):
            if os.path.exists(tmp):
                os.remove(tmp)
        if os.path.exists(path):
            return path
        raise DataUnavailableError(f"Could not download or find cached data for {fmt}.")

    def fetch_venue_text(self, venue_name):
        """Wikipedia summary for a ground ('' if not found). Tries the full name, then the part before the comma."""
        file_path = os.path.join(self.cache_dir, re.sub(r"[^\w\-]+", "_", venue_name) + ".txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        for candidate in dict.fromkeys([venue_name, venue_name.split(",")[0].strip()]):
            try:
                r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                                 + requests.utils.quote(candidate.replace(" ", "_")), timeout=5)
                if r.status_code == 200:
                    extract = r.json().get("extract", "")
                    if extract:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(extract)
                        return extract
            except (requests.RequestException, OSError, ValueError):
                continue
        return ""

    def load_cached_data(self, csv_path):
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        raise DataUnavailableError(f"Cached processed dataset not found at {csv_path}.")
