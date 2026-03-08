"""Train and serve win-probability models from cleaned NFL play-by-play data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency is environment-specific
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pbp_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "win_probability.joblib"

FEATURE_COLUMNS = [
    "score_differential",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]
TARGET_COLUMN = "win"

MODEL_ARTIFACT: dict[str, Any] | None = None


def load_clean_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned play-by-play dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. Run data_loader.py first."
        )

    df = pd.read_csv(data_path, low_memory=False)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "game_id"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in cleaned dataset: {missing}")

    return df


def _prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract features, target, and group ids for split-by-game."""
    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN, "game_id"]].dropna(
        subset=[TARGET_COLUMN, "game_id"]
    )

    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN].astype(int)
    groups = model_df["game_id"].astype(str)

    return X, y, groups


def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def _build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _build_xgboost_pipeline() -> Pipeline:
    if XGBClassifier is None:
        extra_hint = (
            " On macOS, install OpenMP with `brew install libomp`."
            if XGBOOST_IMPORT_ERROR is not None
            else ""
        )
        raise ImportError(
            "xgboost is unavailable in this environment."
            " Install/repair it (e.g., `pip install xgboost`)."
            f"{extra_hint}"
        )

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )

    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", xgb_model)])


def _evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }


def _pick_best_model(metrics: dict[str, dict[str, float]]) -> str:
    ranked = sorted(
        metrics.items(),
        key=lambda item: (item[1]["roc_auc"], -item[1]["brier_score"]),
        reverse=True,
    )
    return ranked[0][0]


def train_win_probability_models(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Train logistic regression and XGBoost models and persist them."""
    df = load_clean_dataset(data_path)
    X, y, groups = _prepare_model_data(df)
    X_train, X_test, y_train, y_test = _split_train_test(X, y, groups)

    logistic_model = _build_logistic_pipeline()
    logistic_model.fit(X_train, y_train)

    xgboost_model = _build_xgboost_pipeline()
    xgboost_model.fit(X_train, y_train)

    metrics = {
        "logistic_regression": _evaluate_model(logistic_model, X_test, y_test),
        "xgboost": _evaluate_model(xgboost_model, X_test, y_test),
    }
    best_model = _pick_best_model(metrics)

    artifact = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "logistic_regression": logistic_model,
        "xgboost": xgboost_model,
        "metrics": metrics,
        "best_model": best_model,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    global MODEL_ARTIFACT
    MODEL_ARTIFACT = artifact

    return artifact


def _load_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    global MODEL_ARTIFACT

    if MODEL_ARTIFACT is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run train_win_probability_models first."
            )
        MODEL_ARTIFACT = joblib.load(model_path)

    return MODEL_ARTIFACT


def predict_win_probability(game_state_dict: dict[str, Any]) -> float:
    """Return offense win probability for a single game-state dictionary."""
    artifact = _load_artifact()

    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in game_state_dict]
    if missing_features:
        raise KeyError(f"Missing features in game_state_dict: {missing_features}")

    model_name = artifact["best_model"]
    model = artifact[model_name]

    input_row = {feature: game_state_dict[feature] for feature in FEATURE_COLUMNS}
    input_df = pd.DataFrame([input_row])

    probability = float(model.predict_proba(input_df)[0, 1])
    return probability


def print_evaluation(metrics: dict[str, dict[str, float]]) -> None:
    """Print ROC AUC and Brier score for each model."""
    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name}: ROC AUC={model_metrics['roc_auc']:.4f}, "
            f"Brier={model_metrics['brier_score']:.4f}"
        )


if __name__ == "__main__":
    trained = train_win_probability_models()
    print_evaluation(trained["metrics"])
    print(f"Best model: {trained['best_model']}")
    print(f"Saved model artifact to {MODEL_PATH}")
