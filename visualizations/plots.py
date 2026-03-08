"""Visualization suite for NFL analytics report charts."""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
from typing import Any

# Avoid matplotlib cache issues in environments where ~/.matplotlib is not writable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fourth_down_engine import recommend_4th_down_decision
from game_simulator import TeamProfile, simulate_single_game
from win_probability_model import FEATURE_COLUMNS, predict_win_probability

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"
QB_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qb_efficiency.csv"
OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "report"


def _load_pbp_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_loader.py first.")
    return pd.read_csv(data_path, low_memory=False)


def _build_qb_efficiency_df(pbp: pd.DataFrame, min_dropbacks: int = 200) -> pd.DataFrame:
    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["passer_player_name"].notna()].copy()

    qb_df = (
        pass_plays.groupby("passer_player_name")
        .agg(
            total_dropbacks=("passer_player_name", "size"),
            EPA_per_play=("epa", "mean"),
            CPOE=("cpoe", "mean"),
            AirYards=("air_yards", "mean"),
        )
        .reset_index()
        .rename(columns={"passer_player_name": "QB"})
    )

    qb_df = qb_df[qb_df["total_dropbacks"] >= min_dropbacks].copy()
    qb_df = qb_df[["QB", "EPA_per_play", "CPOE", "AirYards", "total_dropbacks"]]
    qb_df = qb_df.sort_values("EPA_per_play", ascending=False).reset_index(drop=True)
    return qb_df


def load_or_create_qb_efficiency_df() -> pd.DataFrame:
    if QB_DATA_PATH.exists():
        qb_df = pd.read_csv(QB_DATA_PATH)
        required = {"QB", "EPA_per_play", "CPOE", "AirYards"}
        if required.issubset(qb_df.columns):
            return qb_df

    pbp = _load_pbp_data()
    qb_df = _build_qb_efficiency_df(pbp)
    QB_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    qb_df[["QB", "EPA_per_play", "CPOE", "AirYards"]].to_csv(QB_DATA_PATH, index=False)
    return qb_df


def plot_qb_epa_vs_cpoe(output_path: Path) -> Path:
    """Chart 1: Scatter plot of EPA vs CPOE for quarterbacks."""
    qb_df = load_or_create_qb_efficiency_df().dropna(subset=["EPA_per_play", "CPOE"])
    if qb_df.empty:
        raise ValueError("No QB data available for EPA vs CPOE plot.")

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        qb_df["CPOE"],
        qb_df["EPA_per_play"],
        c=qb_df["AirYards"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.35,
    )

    for _, row in qb_df.nlargest(10, "EPA_per_play").iterrows():
        ax.annotate(
            row["QB"],
            (row["CPOE"], row["EPA_per_play"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )

    ax.set_title("Quarterback Efficiency: EPA per Play vs CPOE")
    ax.set_xlabel("Average CPOE")
    ax.set_ylabel("EPA per Play")
    ax.grid(alpha=0.25)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average Air Yards")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_fourth_down_decision_heatmap(
    output_path: Path,
    base_state: dict[str, Any] | None = None,
) -> Path:
    """Chart 2: Heatmap of optimal fourth-down decisions."""
    if base_state is None:
        base_state = {
            "score_differential": 0,
            "game_seconds_remaining": 600,
            "posteam_timeouts_remaining": 2,
            "defteam_timeouts_remaining": 2,
        }

    yardlines = np.arange(5, 96, 5)
    ydstogo_values = np.arange(1, 16)

    decision_to_code = {"PUNT": 0, "FIELD_GOAL": 1, "GO_FOR_IT": 2}
    code_matrix = np.zeros((len(ydstogo_values), len(yardlines)))

    for i, ydstogo in enumerate(ydstogo_values):
        for j, yardline in enumerate(yardlines):
            game_state = {
                **base_state,
                "down": 4,
                "ydstogo": int(ydstogo),
                "yardline_100": int(yardline),
            }
            decision, _ = recommend_4th_down_decision(game_state)
            code_matrix[i, j] = decision_to_code[decision]

    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(code_matrix, aspect="auto", cmap=cmap, origin="lower")

    ax.set_title("Optimal 4th-Down Decision Heatmap")
    ax.set_xlabel("Yardline_100 (Distance to End Zone)")
    ax.set_ylabel("Yards To Go")

    ax.set_xticks(np.arange(len(yardlines)))
    ax.set_xticklabels(yardlines)
    ax.set_yticks(np.arange(len(ydstogo_values)))
    ax.set_yticklabels(ydstogo_values)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["PUNT", "FIELD_GOAL", "GO_FOR_IT"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_win_probability_during_game(
    output_path: Path,
    game_id: str | None = None,
) -> Path:
    """Chart 3: Line chart showing win probability during a game."""
    pbp = _load_pbp_data()
    valid = pbp.dropna(subset=["game_id", "posteam", "game_seconds_remaining"]).copy()
    if valid.empty:
        raise ValueError("No valid play rows found for win probability chart.")

    if game_id is None:
        game_id = str(valid["game_id"].iloc[0])

    game_df = valid[valid["game_id"] == game_id].copy()
    if game_df.empty:
        raise ValueError(f"No rows found for game_id={game_id}")

    missing_features = [col for col in FEATURE_COLUMNS if col not in game_df.columns]
    if missing_features:
        raise KeyError(f"Missing model feature columns in game data: {missing_features}")

    game_df = game_df.dropna(subset=FEATURE_COLUMNS + ["posteam", "game_seconds_remaining"]).copy()
    if game_df.empty:
        raise ValueError("No playable rows with complete model features for selected game.")

    reference_team = str(game_df["posteam"].iloc[0])

    wp_values: list[float] = []
    elapsed_seconds: list[float] = []

    for _, row in game_df.iterrows():
        state = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        offense_wp = float(predict_win_probability(state))
        reference_wp = offense_wp if row["posteam"] == reference_team else 1.0 - offense_wp

        wp_values.append(reference_wp)
        elapsed_seconds.append(3600.0 - float(row["game_seconds_remaining"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(elapsed_seconds, wp_values, color="#2c7fb8", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"Win Probability During Game ({game_id})")
    ax.set_xlabel("Elapsed Game Time (seconds)")
    ax.set_ylabel(f"{reference_team} Win Probability")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_simulated_game_score_histogram(
    output_path: Path,
    n_games: int = 10000,
) -> Path:
    """Chart 4: Histogram of simulated game scores."""
    team_a = TeamProfile(
        name="Team A",
        offensive_epa_per_play=0.07,
        defensive_epa_allowed=0.01,
        average_drive_length=155,
        play_success_probabilities={"run": 0.45, "pass": 0.52},
    )
    team_b = TeamProfile(
        name="Team B",
        offensive_epa_per_play=0.02,
        defensive_epa_allowed=-0.03,
        average_drive_length=150,
        play_success_probabilities={"run": 0.43, "pass": 0.49},
    )

    random.seed(42)
    team_a_scores: list[int] = []
    team_b_scores: list[int] = []

    for _ in range(n_games):
        score_a, score_b = simulate_single_game(team_a, team_b)
        team_a_scores.append(score_a)
        team_b_scores.append(score_b)

    max_score = int(max(max(team_a_scores), max(team_b_scores)))
    bins = np.arange(-0.5, max_score + 1.5, 1)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(team_a_scores, bins=bins, alpha=0.6, label=team_a.name, color="#1f77b4")
    ax.hist(team_b_scores, bins=bins, alpha=0.6, label=team_b.name, color="#ff7f0e")

    ax.set_title(f"Histogram of Simulated Game Scores ({n_games:,} Simulations)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def create_nfl_analytics_report_visualizations() -> dict[str, Path]:
    """Generate all report charts and return output paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "qb_scatter": plot_qb_epa_vs_cpoe(OUTPUT_DIR / "01_qb_epa_vs_cpoe.png"),
        "fourth_down_heatmap": plot_fourth_down_decision_heatmap(
            OUTPUT_DIR / "02_fourth_down_decision_heatmap.png"
        ),
        "win_probability_line": plot_win_probability_during_game(
            OUTPUT_DIR / "03_win_probability_during_game.png"
        ),
        "simulated_scores_histogram": plot_simulated_game_score_histogram(
            OUTPUT_DIR / "04_simulated_scores_histogram.png"
        ),
    }
    return outputs


if __name__ == "__main__":
    chart_paths = create_nfl_analytics_report_visualizations()
    for chart_name, path in chart_paths.items():
        print(f"{chart_name}: {path}")
