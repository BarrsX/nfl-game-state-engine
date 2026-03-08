"""QB efficiency model using EPA and CPOE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PBP_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"
QB_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "qb_efficiency.csv"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "visualizations" / "qb_efficiency_scatter.png"


def load_play_by_play_data(pbp_path: Path = PBP_PATH) -> pd.DataFrame:
    """Load cleaned play-by-play dataset."""
    if not pbp_path.exists():
        raise FileNotFoundError(
            f"Play-by-play dataset not found at {pbp_path}. Run data_loader.py first."
        )

    df = pd.read_csv(pbp_path, low_memory=False)
    required_columns = ["play_type", "passer_player_name", "epa", "cpoe", "air_yards"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    return df


def filter_pass_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to pass plays with a known passer."""
    pass_plays = df[(df["play_type"] == "pass") & (df["passer_player_name"].notna())].copy()
    return pass_plays


def build_qb_efficiency_dataframe(
    pass_plays: pd.DataFrame,
    min_dropbacks: int = 200,
) -> pd.DataFrame:
    """Aggregate QB efficiency metrics and apply a minimum dropback threshold."""
    qb_stats = (
        pass_plays.groupby("passer_player_name", dropna=False)
        .agg(
            total_dropbacks=("passer_player_name", "size"),
            EPA_per_play=("epa", "mean"),
            CPOE=("cpoe", "mean"),
            AirYards=("air_yards", "mean"),
        )
        .reset_index()
        .rename(columns={"passer_player_name": "QB"})
    )

    qb_stats = qb_stats[qb_stats["total_dropbacks"] >= min_dropbacks].copy()
    qb_stats = qb_stats[["QB", "EPA_per_play", "CPOE", "AirYards", "total_dropbacks"]]
    qb_stats = qb_stats.sort_values("EPA_per_play", ascending=False).reset_index(drop=True)

    return qb_stats


def create_qb_efficiency_scatter_plot(
    qb_df: pd.DataFrame,
    output_path: Path = PLOT_OUTPUT_PATH,
    label_top_n: int = 10,
) -> Path:
    """Create EPA vs CPOE scatter plot and label top quarterbacks by EPA/play."""
    if qb_df.empty:
        raise ValueError("No quarterbacks available after filtering. Check input data/threshold.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        qb_df["CPOE"],
        qb_df["EPA_per_play"],
        c=qb_df["AirYards"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )

    top_qbs = qb_df.nlargest(label_top_n, "EPA_per_play")
    for _, row in top_qbs.iterrows():
        ax.annotate(
            row["QB"],
            (row["CPOE"], row["EPA_per_play"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )

    ax.set_title("QB Efficiency: EPA per Play vs CPOE")
    ax.set_xlabel("CPOE")
    ax.set_ylabel("EPA per Play")
    ax.grid(True, alpha=0.25)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Average Air Yards")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def run_qb_efficiency_model(
    pbp_path: Path = PBP_PATH,
    min_dropbacks: int = 200,
    qb_output_path: Path = QB_OUTPUT_PATH,
    plot_output_path: Path = PLOT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Run the full QB efficiency workflow."""
    pbp = load_play_by_play_data(pbp_path)
    pass_plays = filter_pass_plays(pbp)
    qb_df = build_qb_efficiency_dataframe(pass_plays, min_dropbacks=min_dropbacks)

    qb_output_path.parent.mkdir(parents=True, exist_ok=True)
    qb_df[["QB", "EPA_per_play", "CPOE", "AirYards"]].to_csv(qb_output_path, index=False)

    saved_plot = create_qb_efficiency_scatter_plot(qb_df, output_path=plot_output_path)
    return qb_df, saved_plot


if __name__ == "__main__":
    qb_efficiency_df, plot_path = run_qb_efficiency_model()
    print(qb_efficiency_df[["QB", "EPA_per_play", "CPOE", "AirYards"]].head(10).to_string(index=False))
    print(f"Saved QB efficiency dataset to {QB_OUTPUT_PATH}")
    print(f"Saved scatter plot to {plot_path}")
