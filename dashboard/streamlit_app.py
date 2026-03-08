"""Interactive Streamlit dashboard for NFL game-state analytics."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import streamlit as st

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fourth_down_engine import recommend_4th_down_decision
from game_simulator import TeamProfile, simulate_matchup
from win_probability_model import FEATURE_COLUMNS, MODEL_PATH, predict_win_probability

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"


@st.cache_data(show_spinner=False)
def load_pbp_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
    return pd.read_csv(data_path, low_memory=False)


@st.cache_data(show_spinner=False)
def build_qb_efficiency_df(min_dropbacks: int = 200) -> pd.DataFrame:
    pbp = load_pbp_data()
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
    qb_df = qb_df.sort_values("EPA_per_play", ascending=False).reset_index(drop=True)
    return qb_df


@st.cache_data(show_spinner=False)
def build_fourth_down_heatmap(
    score_differential: int,
    game_seconds_remaining: int,
    posteam_timeouts_remaining: int,
    defteam_timeouts_remaining: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yardlines = np.arange(5, 96, 5)
    ydstogo_values = np.arange(1, 16)

    decision_to_code = {"PUNT": 0, "FIELD_GOAL": 1, "GO_FOR_IT": 2}
    code_matrix = np.zeros((len(ydstogo_values), len(yardlines)))

    for i, ydstogo in enumerate(ydstogo_values):
        for j, yardline in enumerate(yardlines):
            game_state = {
                "score_differential": score_differential,
                "game_seconds_remaining": game_seconds_remaining,
                "down": 4,
                "ydstogo": int(ydstogo),
                "yardline_100": int(yardline),
                "posteam_timeouts_remaining": posteam_timeouts_remaining,
                "defteam_timeouts_remaining": defteam_timeouts_remaining,
            }
            decision, _ = recommend_4th_down_decision(game_state)
            code_matrix[i, j] = decision_to_code[decision]

    return code_matrix, yardlines, ydstogo_values


@st.cache_data(show_spinner=False)
def build_win_probability_series(game_id: str) -> pd.DataFrame:
    pbp = load_pbp_data()
    game_df = pbp[pbp["game_id"] == game_id].copy()

    needed = FEATURE_COLUMNS + ["posteam", "game_seconds_remaining"]
    game_df = game_df.dropna(subset=needed).copy()
    if game_df.empty:
        raise ValueError(f"No valid rows with features found for game {game_id}.")

    game_df = game_df.sort_values("game_seconds_remaining", ascending=False).reset_index(drop=True)
    game_length = float(game_df["game_seconds_remaining"].max())
    reference_team = str(game_df["posteam"].iloc[0])

    elapsed_minutes: list[float] = []
    reference_team_wp: list[float] = []

    for _, row in game_df.iterrows():
        state = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        offense_wp = float(predict_win_probability(state))
        ref_wp = offense_wp if row["posteam"] == reference_team else 1.0 - offense_wp

        elapsed_minutes.append((game_length - float(row["game_seconds_remaining"])) / 60.0)
        reference_team_wp.append(ref_wp)

    return pd.DataFrame(
        {
            "elapsed_minutes": elapsed_minutes,
            "reference_team_wp": reference_team_wp,
            "reference_team": reference_team,
        }
    )


def render_overview() -> None:
    pbp = load_pbp_data()

    season_prefixes = sorted({str(gid)[:4] for gid in pbp["game_id"].dropna().unique()})
    games = int(pbp["game_id"].nunique())
    plays = int(len(pbp))
    pass_plays = int((pbp["play_type"] == "pass").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{plays:,}")
    m2.metric("Games", f"{games:,}")
    m3.metric("Pass Plays", f"{pass_plays:,}")
    m4.metric("Seasons", ", ".join(season_prefixes) if season_prefixes else "N/A")

    st.info(
        "Use the sidebar to explore QB efficiency, fourth-down recommendations, in-game win probability, "
        "and Monte Carlo simulation outputs."
    )



def render_qb_scatter() -> None:
    min_dropbacks = st.slider("Minimum Dropbacks", min_value=50, max_value=400, value=200, step=10)
    qb_df = build_qb_efficiency_df(min_dropbacks=min_dropbacks)

    if qb_df.empty:
        st.warning("No QBs meet the current dropback threshold.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        qb_df["CPOE"],
        qb_df["EPA_per_play"],
        c=qb_df["AirYards"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )

    for _, row in qb_df.nlargest(10, "EPA_per_play").iterrows():
        ax.annotate(
            row["QB"],
            (row["CPOE"], row["EPA_per_play"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
        )

    ax.set_title("Quarterback Efficiency: EPA per Play vs CPOE")
    ax.set_xlabel("CPOE")
    ax.set_ylabel("EPA per Play")
    ax.grid(alpha=0.25)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average Air Yards")

    st.pyplot(fig, clear_figure=True)
    st.dataframe(
        qb_df[["QB", "total_dropbacks", "EPA_per_play", "CPOE", "AirYards"]].head(25),
        use_container_width=True,
    )



def render_fourth_down() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        score_diff = st.number_input("Score Differential", min_value=-30, max_value=30, value=0, step=1)
        ydstogo = st.number_input("Yards To Go", min_value=1, max_value=20, value=4, step=1)
    with c2:
        game_seconds = st.number_input(
            "Game Seconds Remaining", min_value=1, max_value=4200, value=600, step=1
        )
        yardline_100 = st.number_input(
            "Yardline_100 (distance to end zone)", min_value=1, max_value=99, value=45, step=1
        )
    with c3:
        off_to = st.number_input("Offense Timeouts", min_value=0, max_value=3, value=2, step=1)
        def_to = st.number_input("Defense Timeouts", min_value=0, max_value=3, value=2, step=1)

    if st.button("Recommend 4th Down Decision"):
        game_state = {
            "score_differential": int(score_diff),
            "game_seconds_remaining": int(game_seconds),
            "down": 4,
            "ydstogo": int(ydstogo),
            "yardline_100": int(yardline_100),
            "posteam_timeouts_remaining": int(off_to),
            "defteam_timeouts_remaining": int(def_to),
        }
        try:
            decision, expected_wp = recommend_4th_down_decision(game_state)
            st.success(f"Recommendation: {decision}")
            st.metric("Expected Win Probability", f"{expected_wp:.3f}")
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"Could not compute recommendation: {exc}")

    st.subheader("Optimal Decision Heatmap")
    try:
        code_matrix, yardlines, ydstogo_values = build_fourth_down_heatmap(
            int(score_diff), int(game_seconds), int(off_to), int(def_to)
        )

        cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(code_matrix, aspect="auto", cmap=cmap, origin="lower")

        ax.set_title("Optimal 4th-Down Decision by Field Position and Distance")
        ax.set_xlabel("Yardline_100")
        ax.set_ylabel("Yards To Go")

        ax.set_xticks(np.arange(len(yardlines)))
        ax.set_xticklabels(yardlines)
        ax.set_yticks(np.arange(len(ydstogo_values)))
        ax.set_yticklabels(ydstogo_values)

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["PUNT", "FIELD_GOAL", "GO_FOR_IT"])

        st.pyplot(fig, clear_figure=True)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Could not build heatmap: {exc}")



def render_win_probability_timeline() -> None:
    if not MODEL_PATH.exists():
        st.error(
            f"Win probability model artifact not found at {MODEL_PATH}. "
            "Run src/win_probability_model.py first."
        )
        return

    pbp = load_pbp_data()
    game_ids = sorted(pbp["game_id"].dropna().astype(str).unique())
    if not game_ids:
        st.warning("No game IDs available in dataset.")
        return

    game_id = st.selectbox("Select Game ID", options=game_ids, index=0)

    try:
        wp_df = build_win_probability_series(game_id)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Could not compute win probability timeline: {exc}")
        return

    reference_team = str(wp_df["reference_team"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(wp_df["elapsed_minutes"], wp_df["reference_team_wp"], color="#2c7fb8", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"{reference_team} Win Probability During {game_id}")
    ax.set_xlabel("Elapsed Game Minutes")
    ax.set_ylabel("Win Probability")
    ax.grid(alpha=0.3)

    st.pyplot(fig, clear_figure=True)
    st.metric(
        label=f"Final Predicted Win Probability ({reference_team})",
        value=f"{wp_df['reference_team_wp'].iloc[-1]:.3f}",
    )



def render_simulation() -> None:
    st.caption("Simulate matchup outcomes with custom efficiency inputs.")

    col_a, col_b = st.columns(2)
    with col_a:
        team_a_name = st.text_input("Team A Name", value="Team A")
        team_a_off_epa = st.slider("Team A Offensive EPA/Play", -0.3, 0.3, 0.07, 0.01)
        team_a_def_epa = st.slider("Team A Defensive EPA Allowed", -0.3, 0.3, 0.01, 0.01)
        team_a_drive = st.slider("Team A Average Drive Length (sec)", 90, 240, 155, 5)
        team_a_run_success = st.slider("Team A Run Success Probability", 0.20, 0.70, 0.45, 0.01)
        team_a_pass_success = st.slider("Team A Pass Success Probability", 0.20, 0.80, 0.52, 0.01)

    with col_b:
        team_b_name = st.text_input("Team B Name", value="Team B")
        team_b_off_epa = st.slider("Team B Offensive EPA/Play", -0.3, 0.3, 0.02, 0.01)
        team_b_def_epa = st.slider("Team B Defensive EPA Allowed", -0.3, 0.3, -0.03, 0.01)
        team_b_drive = st.slider("Team B Average Drive Length (sec)", 90, 240, 150, 5)
        team_b_run_success = st.slider("Team B Run Success Probability", 0.20, 0.70, 0.43, 0.01)
        team_b_pass_success = st.slider("Team B Pass Success Probability", 0.20, 0.80, 0.49, 0.01)

    sims = st.slider("Number of Simulations", min_value=100, max_value=20000, value=10000, step=100)
    seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

    if st.button("Run Monte Carlo Simulation"):
        team_a = TeamProfile(
            name=team_a_name,
            offensive_epa_per_play=float(team_a_off_epa),
            defensive_epa_allowed=float(team_a_def_epa),
            average_drive_length=float(team_a_drive),
            play_success_probabilities={"run": float(team_a_run_success), "pass": float(team_a_pass_success)},
        )
        team_b = TeamProfile(
            name=team_b_name,
            offensive_epa_per_play=float(team_b_off_epa),
            defensive_epa_allowed=float(team_b_def_epa),
            average_drive_length=float(team_b_drive),
            play_success_probabilities={"run": float(team_b_run_success), "pass": float(team_b_pass_success)},
        )

        with st.spinner("Running simulations..."):
            summary = simulate_matchup(team_a, team_b, n_games=int(sims), seed=int(seed))

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a_name} Win Prob", f"{summary['win_probability_team_a']:.3f}")
        m2.metric("Tie Prob", f"{summary['tie_probability']:.3f}")
        m3.metric(
            "Average Score",
            f"{summary['average_score'][team_a_name]:.1f} - {summary['average_score'][team_b_name]:.1f}",
        )

        score_dist = summary["score_distribution"]

        team_a_counts: Counter[int] = Counter()
        team_b_counts: Counter[int] = Counter()
        for (a_score, b_score), count in score_dist.items():
            team_a_counts[int(a_score)] += int(count)
            team_b_counts[int(b_score)] += int(count)

        x_scores = np.array(sorted(set(team_a_counts) | set(team_b_counts)))
        y_a = np.array([team_a_counts.get(int(score), 0) for score in x_scores])
        y_b = np.array([team_b_counts.get(int(score), 0) for score in x_scores])

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(x_scores - 0.2, y_a, width=0.4, label=team_a_name, alpha=0.75)
        ax.bar(x_scores + 0.2, y_b, width=0.4, label=team_b_name, alpha=0.75)
        ax.set_title("Histogram of Simulated Team Scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

        top_scores = sorted(score_dist.items(), key=lambda kv: kv[1], reverse=True)[:20]
        top_df = pd.DataFrame(
            {
                "Score": [f"{a}-{b}" for (a, b), _ in top_scores],
                "Count": [count for _, count in top_scores],
            }
        )
        st.subheader("Top Score Outcomes")
        st.dataframe(top_df, use_container_width=True)



def main() -> None:
    st.set_page_config(page_title="NFL Game State Engine", layout="wide")
    st.title("NFL Game State Engine Dashboard")

    section = st.sidebar.selectbox(
        "Section",
        [
            "Overview",
            "QB EPA vs CPOE",
            "Fourth-Down Decision Engine",
            "Win Probability Timeline",
            "Monte Carlo Game Simulator",
        ],
    )

    try:
        if section == "Overview":
            render_overview()
        elif section == "QB EPA vs CPOE":
            render_qb_scatter()
        elif section == "Fourth-Down Decision Engine":
            render_fourth_down()
        elif section == "Win Probability Timeline":
            render_win_probability_timeline()
        elif section == "Monte Carlo Game Simulator":
            render_simulation()
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
