"""Fourth-down decision engine powered by win-probability estimates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from win_probability_model import predict_win_probability

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PBP_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"

REQUIRED_STATE_KEYS = [
    "score_differential",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


@dataclass
class DecisionResult:
    decision: str
    expected_win_probability: float


class FourthDownDecisionEngine:
    """Build priors from historical fourth-down plays and recommend decisions."""

    def __init__(self, pbp_path: Path = PBP_PATH) -> None:
        self.pbp_path = pbp_path
        self.pbp = self._load_dataset()
        self.fourth_down_plays = self.identify_fourth_down_plays(self.pbp)

        self.conversion_by_ydstogo, self.global_conversion_rate = self._fit_conversion_model()
        self.fg_by_distance_bin, self.global_fg_rate = self._fit_field_goal_model()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.pbp_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.pbp_path}. Run data_loader.py first."
            )

        df = pd.read_csv(self.pbp_path, low_memory=False)
        required = [
            "down",
            "ydstogo",
            "yardline_100",
            "play_type",
            "epa",
            "score_differential",
            "game_seconds_remaining",
            "posteam_timeouts_remaining",
            "defteam_timeouts_remaining",
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in dataset: {missing}")

        return df

    @staticmethod
    def identify_fourth_down_plays(df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Identify all 4th down plays."""
        return df[df["down"] == 4].copy()

    def _fit_conversion_model(self) -> tuple[pd.DataFrame, float]:
        """Step 2: Estimate 4th-down conversion success by yards-to-go (EPA proxy)."""
        attempts = self.fourth_down_plays[
            self.fourth_down_plays["play_type"].isin(["run", "pass"])
        ].copy()

        if attempts.empty:
            empty = pd.DataFrame(columns=["ydstogo_int", "attempts", "conversions", "p_success"])
            return empty, 0.45

        attempts["ydstogo_int"] = attempts["ydstogo"].clip(lower=1, upper=20).round().astype(int)
        attempts["converted"] = (attempts["epa"] > 0).astype(int)

        by_distance = (
            attempts.groupby("ydstogo_int", as_index=False)
            .agg(attempts=("converted", "size"), conversions=("converted", "sum"))
        )

        global_rate = float(attempts["converted"].mean())
        prior_weight = 30.0
        by_distance["p_success"] = (
            by_distance["conversions"] + prior_weight * global_rate
        ) / (by_distance["attempts"] + prior_weight)

        return by_distance, global_rate

    def _fit_field_goal_model(self) -> tuple[pd.DataFrame, float]:
        """Estimate FG make probability by distance bins (EPA proxy)."""
        kicks = self.fourth_down_plays[self.fourth_down_plays["play_type"] == "field_goal"].copy()

        if kicks.empty:
            empty = pd.DataFrame(columns=["distance_bin", "attempts", "made", "p_success"])
            return empty, 0.78

        kicks["kick_distance"] = (kicks["yardline_100"] + 17).round().astype(int)
        kicks["distance_bin"] = (5 * np.round(kicks["kick_distance"] / 5)).astype(int)
        kicks["made"] = (kicks["epa"] > 0).astype(int)

        by_distance = (
            kicks.groupby("distance_bin", as_index=False)
            .agg(attempts=("made", "size"), made=("made", "sum"))
        )

        global_rate = float(kicks["made"].mean())
        prior_weight = 25.0
        by_distance["p_success"] = (
            by_distance["made"] + prior_weight * global_rate
        ) / (by_distance["attempts"] + prior_weight)

        return by_distance, global_rate

    def estimate_conversion_probability(self, ydstogo: float) -> float:
        """Return conversion probability for go-for-it based on yards-to-go."""
        y_int = int(np.clip(np.round(ydstogo), 1, 20))

        if self.conversion_by_ydstogo.empty:
            return float(np.clip(0.75 - 0.045 * y_int, 0.08, 0.85))

        matched = self.conversion_by_ydstogo[self.conversion_by_ydstogo["ydstogo_int"] == y_int]
        if not matched.empty:
            return float(matched["p_success"].iloc[0])

        return float(np.clip(self.global_conversion_rate - 0.03 * (y_int - 1), 0.06, 0.85))

    def estimate_fg_success_probability(self, kick_distance: float) -> float:
        """Return FG make probability based on kick distance."""
        distance_bin = int(5 * round(kick_distance / 5))

        if self.fg_by_distance_bin.empty:
            return float(np.clip(1.1 - 0.01 * kick_distance, 0.15, 0.98))

        matched = self.fg_by_distance_bin[self.fg_by_distance_bin["distance_bin"] == distance_bin]
        if not matched.empty:
            return float(matched["p_success"].iloc[0])

        return float(np.clip(self.global_fg_rate - 0.008 * (kick_distance - 35), 0.1, 0.98))

    @staticmethod
    def estimate_punt_net_yards(yardline_100: float) -> float:
        """Estimate net punt yards as a function of field position."""
        if yardline_100 >= 65:
            return 42.0
        if yardline_100 >= 50:
            return 40.0
        if yardline_100 >= 35:
            return 36.0
        if yardline_100 >= 20:
            return 32.0
        return 28.0

    @staticmethod
    def _validate_game_state(game_state: dict[str, Any]) -> None:
        missing = [key for key in REQUIRED_STATE_KEYS if key not in game_state]
        if missing:
            raise KeyError(f"Missing game_state keys: {missing}")
        if int(game_state["down"]) != 4:
            raise ValueError("This engine is built for 4th-down decisions. game_state['down'] must be 4.")

    def _expected_wp_go_for_it(self, game_state: dict[str, Any]) -> float:
        p_conv = self.estimate_conversion_probability(float(game_state["ydstogo"]))

        success_yardline = float(max(1.0, game_state["yardline_100"] - max(1.0, game_state["ydstogo"])))
        success_state = {
            "score_differential": float(game_state["score_differential"]),
            "game_seconds_remaining": float(game_state["game_seconds_remaining"]),
            "down": 1,
            "ydstogo": float(min(10.0, success_yardline)),
            "yardline_100": success_yardline,
            "posteam_timeouts_remaining": float(game_state["posteam_timeouts_remaining"]),
            "defteam_timeouts_remaining": float(game_state["defteam_timeouts_remaining"]),
        }
        wp_success = predict_win_probability(success_state)

        fail_state_for_opponent = {
            "score_differential": float(-game_state["score_differential"]),
            "game_seconds_remaining": float(game_state["game_seconds_remaining"]),
            "down": 1,
            "ydstogo": 10.0,
            "yardline_100": float(np.clip(100.0 - game_state["yardline_100"], 1.0, 99.0)),
            "posteam_timeouts_remaining": float(game_state["defteam_timeouts_remaining"]),
            "defteam_timeouts_remaining": float(game_state["posteam_timeouts_remaining"]),
        }
        wp_fail_original_offense = 1.0 - predict_win_probability(fail_state_for_opponent)

        return float(p_conv * wp_success + (1.0 - p_conv) * wp_fail_original_offense)

    def _expected_wp_punt(self, game_state: dict[str, Any]) -> float:
        net_punt = self.estimate_punt_net_yards(float(game_state["yardline_100"]))
        opponent_yardline_100 = float(
            np.clip(100.0 - game_state["yardline_100"] + net_punt, 1.0, 99.0)
        )

        opponent_state = {
            "score_differential": float(-game_state["score_differential"]),
            "game_seconds_remaining": float(game_state["game_seconds_remaining"]),
            "down": 1,
            "ydstogo": 10.0,
            "yardline_100": opponent_yardline_100,
            "posteam_timeouts_remaining": float(game_state["defteam_timeouts_remaining"]),
            "defteam_timeouts_remaining": float(game_state["posteam_timeouts_remaining"]),
        }

        return float(1.0 - predict_win_probability(opponent_state))

    def _expected_wp_field_goal(self, game_state: dict[str, Any]) -> float:
        kick_distance = float(game_state["yardline_100"] + 17.0)
        p_make = self.estimate_fg_success_probability(kick_distance)

        make_state_for_opponent = {
            "score_differential": float(-(game_state["score_differential"] + 3.0)),
            "game_seconds_remaining": float(game_state["game_seconds_remaining"]),
            "down": 1,
            "ydstogo": 10.0,
            "yardline_100": 75.0,
            "posteam_timeouts_remaining": float(game_state["defteam_timeouts_remaining"]),
            "defteam_timeouts_remaining": float(game_state["posteam_timeouts_remaining"]),
        }
        wp_make_original_offense = 1.0 - predict_win_probability(make_state_for_opponent)

        missed_spot_for_opponent = float(np.clip(100.0 - game_state["yardline_100"] - 7.0, 1.0, 99.0))
        miss_state_for_opponent = {
            "score_differential": float(-game_state["score_differential"]),
            "game_seconds_remaining": float(game_state["game_seconds_remaining"]),
            "down": 1,
            "ydstogo": 10.0,
            "yardline_100": missed_spot_for_opponent,
            "posteam_timeouts_remaining": float(game_state["defteam_timeouts_remaining"]),
            "defteam_timeouts_remaining": float(game_state["posteam_timeouts_remaining"]),
        }
        wp_miss_original_offense = 1.0 - predict_win_probability(miss_state_for_opponent)

        return float(p_make * wp_make_original_offense + (1.0 - p_make) * wp_miss_original_offense)

    def recommend(self, game_state: dict[str, Any]) -> DecisionResult:
        self._validate_game_state(game_state)

        expected = {
            "GO_FOR_IT": self._expected_wp_go_for_it(game_state),
            "PUNT": self._expected_wp_punt(game_state),
            "FIELD_GOAL": self._expected_wp_field_goal(game_state),
        }

        decision = max(expected, key=expected.get)
        return DecisionResult(decision=decision, expected_win_probability=float(expected[decision]))


_ENGINE: FourthDownDecisionEngine | None = None


def _get_engine() -> FourthDownDecisionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FourthDownDecisionEngine()
    return _ENGINE


def recommend_4th_down_decision(game_state: dict[str, Any]) -> tuple[str, float]:
    """Return recommended 4th-down decision and expected win probability."""
    result = _get_engine().recommend(game_state)
    return result.decision, result.expected_win_probability


if __name__ == "__main__":
    sample_state = {
        "score_differential": 2,
        "game_seconds_remaining": 380,
        "down": 4,
        "ydstogo": 3,
        "yardline_100": 46,
        "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 1,
    }

    decision, expected_wp = recommend_4th_down_decision(sample_state)
    print(f"Recommended decision: {decision}")
    print(f"Expected win probability: {expected_wp:.4f}")
