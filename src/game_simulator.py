"""Monte Carlo NFL game simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Any

GAME_SECONDS = 60 * 60
START_YARDLINE_100 = 75.0  # Typical touchback start: own 25-yard line.

# Approximate historical play outcome buckets (yards gained) for successful/failed plays.
RUN_SUCCESS_YARDS = [4, 5, 6, 7, 8, 10, 12]
RUN_FAIL_YARDS = [-3, -2, -1, 0, 1, 2]
PASS_SUCCESS_YARDS = [6, 8, 10, 12, 15, 20, 25, 35]
PASS_FAIL_YARDS = [-10, -7, -5, -2, 0]  # sacks/incompletions/throwaways


@dataclass
class TeamProfile:
    name: str
    offensive_epa_per_play: float
    defensive_epa_allowed: float
    average_drive_length: float  # seconds
    play_success_probabilities: dict[str, float]  # keys: run, pass


@dataclass
class DriveResult:
    outcome: str  # touchdown | field_goal | turnover | punt | clock_expired
    points: int
    seconds_used: float
    next_start_yardline_100: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _opponent_perspective_yardline(yardline_100: float) -> float:
    return _clamp(100.0 - yardline_100, 1.0, 99.0)


def _epa_adjusted_success_prob(
    base_prob: float,
    offense_epa: float,
    opponent_def_epa_allowed: float,
) -> float:
    # Positive offensive EPA and positive opponent EPA-allowed should increase success.
    adjustment = 0.18 * offense_epa + 0.12 * opponent_def_epa_allowed
    return _clamp(base_prob + adjustment, 0.15, 0.9)


def _select_play_type(play_success_probabilities: dict[str, float]) -> str:
    run_weight = float(play_success_probabilities.get("run", 0.48))
    pass_weight = float(play_success_probabilities.get("pass", 0.52))
    total = run_weight + pass_weight
    if total <= 0:
        return "pass"

    run_share = run_weight / total
    return "run" if random.random() < run_share else "pass"


def _sample_play_seconds(target_drive_length: float) -> float:
    # Approximate 5.8 plays/drive around NFL average.
    mean_play_seconds = max(12.0, target_drive_length / 5.8)
    return _clamp(random.gauss(mean_play_seconds, 6.0), 8.0, 45.0)


def _touchdown_on_play(yardline_100: float, yards_gained: float) -> bool:
    return (yardline_100 - yards_gained) <= 0


def _field_goal_make_probability(yardline_100: float) -> float:
    kick_distance = yardline_100 + 17.0
    return _clamp(1.08 - 0.0105 * kick_distance, 0.12, 0.98)


def _punt_net_yards(yardline_100: float) -> float:
    if yardline_100 >= 65:
        mean = 42.0
    elif yardline_100 >= 50:
        mean = 40.0
    elif yardline_100 >= 35:
        mean = 36.0
    else:
        mean = 31.0
    return _clamp(random.gauss(mean, 7.0), 20.0, 55.0)


def _turnover_probability(play_type: str) -> float:
    return 0.026 if play_type == "pass" else 0.012


def simulate_drive(
    offense: TeamProfile,
    defense: TeamProfile,
    start_yardline_100: float = START_YARDLINE_100,
    remaining_seconds: float = GAME_SECONDS,
) -> DriveResult:
    """Simulate one drive with yardline/down/distance state transitions."""
    yardline_100 = _clamp(start_yardline_100, 1.0, 99.0)
    down = 1
    distance = 10.0
    seconds_used = 0.0

    while remaining_seconds - seconds_used > 0:
        play_type = _select_play_type(offense.play_success_probabilities)
        base_success = float(offense.play_success_probabilities.get(play_type, 0.5))
        p_success = _epa_adjusted_success_prob(
            base_success,
            offense.offensive_epa_per_play,
            defense.defensive_epa_allowed,
        )

        success = random.random() < p_success
        if play_type == "run":
            yards = random.choice(RUN_SUCCESS_YARDS if success else RUN_FAIL_YARDS)
        else:
            yards = random.choice(PASS_SUCCESS_YARDS if success else PASS_FAIL_YARDS)

        # Occasional explosive gain tied to stronger offenses.
        explosive_boost = max(0.0, offense.offensive_epa_per_play) * random.uniform(0.0, 8.0)
        yards += explosive_boost

        seconds_used += _sample_play_seconds(offense.average_drive_length)

        if _touchdown_on_play(yardline_100, yards):
            return DriveResult(
                outcome="touchdown",
                points=7,
                seconds_used=seconds_used,
                next_start_yardline_100=START_YARDLINE_100,
            )

        yardline_100 = _clamp(yardline_100 - yards, 1.0, 99.0)

        if random.random() < _turnover_probability(play_type):
            return DriveResult(
                outcome="turnover",
                points=0,
                seconds_used=seconds_used,
                next_start_yardline_100=_opponent_perspective_yardline(yardline_100),
            )

        if yards >= distance:
            down = 1
            distance = min(10.0, yardline_100)
        else:
            distance -= max(yards, 0.0)
            down += 1

        if down > 4:
            # 4th-down decision heuristic during simulation.
            if yardline_100 <= 35:
                make_prob = _field_goal_make_probability(yardline_100)
                made = random.random() < make_prob
                return DriveResult(
                    outcome="field_goal" if made else "turnover",
                    points=3 if made else 0,
                    seconds_used=seconds_used,
                    next_start_yardline_100=START_YARDLINE_100 if made else _opponent_perspective_yardline(yardline_100),
                )

            net_punt = _punt_net_yards(yardline_100)
            opp_start = _clamp(100.0 - yardline_100 + net_punt, 1.0, 99.0)
            return DriveResult(
                outcome="punt",
                points=0,
                seconds_used=seconds_used,
                next_start_yardline_100=opp_start,
            )

    return DriveResult(
        outcome="clock_expired",
        points=0,
        seconds_used=remaining_seconds,
        next_start_yardline_100=START_YARDLINE_100,
    )


def simulate_single_game(team_a: TeamProfile, team_b: TeamProfile) -> tuple[int, int]:
    """Simulate one full game until the clock reaches zero."""
    clock = float(GAME_SECONDS)
    score_a = 0
    score_b = 0

    offense = team_a
    defense = team_b
    offense_is_a = True
    drive_start = START_YARDLINE_100

    while clock > 0:
        drive = simulate_drive(
            offense=offense,
            defense=defense,
            start_yardline_100=drive_start,
            remaining_seconds=clock,
        )
        clock -= drive.seconds_used

        if offense_is_a:
            score_a += drive.points
        else:
            score_b += drive.points

        drive_start = drive.next_start_yardline_100
        offense, defense = defense, offense
        offense_is_a = not offense_is_a

    return score_a, score_b


def simulate_matchup(
    team_a: TeamProfile,
    team_b: TeamProfile,
    n_games: int = 10000,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Simulate many games and return win probability and score summaries."""
    if n_games <= 0:
        raise ValueError("n_games must be greater than zero")

    if seed is not None:
        random.seed(seed)

    score_pairs: list[tuple[int, int]] = []
    wins_a = 0
    ties = 0

    for _ in range(n_games):
        a_score, b_score = simulate_single_game(team_a, team_b)
        score_pairs.append((a_score, b_score))
        if a_score > b_score:
            wins_a += 1
        elif a_score == b_score:
            ties += 1

    distribution = Counter(score_pairs)
    avg_a = sum(a for a, _ in score_pairs) / n_games
    avg_b = sum(b for _, b in score_pairs) / n_games

    return {
        "team_a": team_a.name,
        "team_b": team_b.name,
        "simulations": n_games,
        "win_probability_team_a": wins_a / n_games,
        "tie_probability": ties / n_games,
        "score_distribution": dict(distribution),
        "average_score": {team_a.name: avg_a, team_b.name: avg_b},
    }


if __name__ == "__main__":
    home = TeamProfile(
        name="Team A",
        offensive_epa_per_play=0.07,
        defensive_epa_allowed=0.01,
        average_drive_length=155,
        play_success_probabilities={"run": 0.45, "pass": 0.52},
    )
    away = TeamProfile(
        name="Team B",
        offensive_epa_per_play=0.02,
        defensive_epa_allowed=-0.03,
        average_drive_length=150,
        play_success_probabilities={"run": 0.43, "pass": 0.49},
    )

    summary = simulate_matchup(home, away, n_games=10000, seed=7)
    print(f"{summary['team_a']} win probability: {summary['win_probability_team_a']:.3f}")
    print(f"Average score: {summary['average_score']}")
    print("Top score outcomes (Team A, Team B):")
    for score, count in summary["score_distribution"].items():
        print(f"  {score}: {count}")
