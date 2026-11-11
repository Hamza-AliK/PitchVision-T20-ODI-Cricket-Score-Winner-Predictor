# PitchVision: T20 / ODI Cricket Score & Winner Predictor

**PitchVision** is a Python console application designed to forecast the outcome of men's international T20 and ODI cricket matches using historical ball-by-ball match data and venue descriptions. Built as a laboratory project for **AL-2001 Programming for Artificial Intelligence (Fall 2026)** at **FAST-NUCES Karachi Campus**.

---

## 📌 Project Overview
 Cricket score predictions often rely on informal run-rate arithmetic, which ignores crucial context such as wickets remaining, pitch conditions, and historical venue behavior. PitchVision solves this by:
1. Cleaning and parsing thousands of past T20 & ODI matches.
2. Engineering statistical match features and using date-based data splits to evaluate predictive performance honestly.
3. Leveraging Natural Language Processing (NLP - Track A) to extract pitch characteristics from Wikipedia ground descriptions and rank similar venues using TF-IDF and Cosine Similarity.
4. Supplying a robust, crash-free user interface with offline fallback capability.

---

## 👥 Group Information & Division of Work

| Name | Roll Number | Section | Role & Modules Owned | Work Share |
| :--- | :--- | :--- | :--- | :--- |
| **Hamza Ali Khatri** *(Leader)* | 25K-0004 | BAI-3A | `data_manager.py`, `exceptions.py`, `main.py` (ConsoleMenu), repository setup, caching, offline fallback | **34%** |
| **Abdul Rehman** | 25K-0062 | BAI-3A | `preprocessing.py` (Pandas cleaning, feature engineering, NumPy over matrix), `visualiser.py` (Matplotlib charts) | **33%** |
| **Syed M. Abbas** | 25K-0023 | BAI-3A | `models.py` (BasePredictor, ScorePredictor, WinnerPredictor, polymorphism), `text_processor.py` (Track A NLP, TF-IDF, similarity) | **33%** |

---

## 🏗️ Project Architecture & Directory Layout

```text
PitchVision-T20-ODI-Cricket-Score-Winner-Predictor/
│
├── data/
│   └── raw/                    # Cached raw match files & Wikipedia text summaries
│
├── outputs/
│   └── figures/                # Saved Matplotlib/Seaborn visualization plots
│
├── exceptions.py               # Custom exception classes (DataUnavailableError)
├── data_manager.py             # Data fetching, offline fallback, and local caching
├── preprocessing.py            # Pandas cleaning, feature building, date splitting, NumPy matrix
├── text_processor.py           # NLP cleaning, stop-word removal, TF-IDF, Cosine similarity
├── models.py                   # Abstract BasePredictor, ScorePredictor, WinnerPredictor
├── visualiser.py               # Matplotlib visualization pipeline
├── main.py                     # Interactive Console Menu (Application Entry Point)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
