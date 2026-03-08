"""Load and clean NFL play-by-play data."""

from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl

# NFL "2025-26 season" is season=2025 in nfl_data_py (regular + postseason).
SEASONS = [2025]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"

COLUMNS_TO_KEEP = [
    "game_id",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "play_type",
    "epa",
    "air_yards",
    "complete_pass",
    "cp",
    "cpoe",
    "passer_player_name",
    "passer_player_id",
]


def load_pbp_data(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Download play-by-play data for the specified seasons."""
    return nfl.import_pbp_data(seasons)


def remove_unwanted_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Remove kneels, spikes, and penalties."""
    drop_mask = pd.Series(False, index=df.index)

    if "qb_kneel" in df.columns:
        drop_mask |= df["qb_kneel"].fillna(0).astype(int).eq(1)
    if "qb_spike" in df.columns:
        drop_mask |= df["qb_spike"].fillna(0).astype(int).eq(1)
    if "penalty" in df.columns:
        drop_mask |= df["penalty"].fillna(0).astype(int).eq(1)
    if "play_type" in df.columns:
        drop_mask |= df["play_type"].isin(["qb_kneel", "qb_spike", "no_play"])

    return df.loc[~drop_mask].copy()


def _select_score_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Pick score columns available in nfl_data_py PBP schema."""
    candidate_pairs = [
        ("home_score", "away_score"),
        ("total_home_score", "total_away_score"),
    ]

    for home_col, away_col in candidate_pairs:
        if home_col in df.columns and away_col in df.columns:
            return home_col, away_col

    raise KeyError(
        "Could not find home/away score columns in PBP data. "
        "Expected one of: home_score/away_score or total_home_score/total_away_score."
    )


def add_win_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary `win` column: 1 if posteam won game, else 0."""
    home_score_col, away_score_col = _select_score_columns(df)
    required_team_cols = ["home_team", "away_team"]
    missing_team_cols = [col for col in required_team_cols if col not in df.columns]
    if missing_team_cols:
        raise KeyError(f"Missing required team columns in source data: {missing_team_cols}")

    game_results = (
        df.groupby("game_id", as_index=False)
        .agg(
            home_team=("home_team", "last"),
            away_team=("away_team", "last"),
            home_final=(home_score_col, "max"),
            away_final=(away_score_col, "max"),
        )
        .dropna(subset=["home_team", "away_team"])
    )

    game_results["winner"] = np.select(
        [
            game_results["home_final"] > game_results["away_final"],
            game_results["away_final"] > game_results["home_final"],
        ],
        [game_results["home_team"], game_results["away_team"]],
        default=pd.NA,
    )

    out = df.merge(game_results[["game_id", "winner"]], on="game_id", how="left")
    out["win"] = (out["posteam"] == out["winner"]).fillna(False).astype("int8")

    return out.drop(columns=["winner"])


def build_clean_pbp_dataset() -> pd.DataFrame:
    """Run the full pipeline and return cleaned play-by-play data."""
    pbp = load_pbp_data(SEASONS)
    pbp = remove_unwanted_plays(pbp)
    pbp = add_win_column(pbp)

    missing_cols = [col for col in COLUMNS_TO_KEEP if col not in pbp.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in source data: {missing_cols}")

    return pbp[COLUMNS_TO_KEEP + ["win"]].copy()


def save_clean_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    """Build and save the cleaned dataset."""
    cleaned = build_clean_pbp_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    destination = save_clean_dataset()
    print(f"Saved cleaned dataset to {destination}")
