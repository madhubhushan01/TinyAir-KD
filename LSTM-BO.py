"""
Generic LSTM-BO pipeline for hourly PM2.5 forecasting.

Protocol
--------
Input: 8 consecutive hourly observations [t-7, ..., t]
Target: PM2.5 at the immediate next hour t+1
Split: chronological 70% train, 15% validation, 15% test
Selection: Bayesian optimization using validation R² only
Evaluation: test set is used only after hyperparameter selection

Exact LSTM-BO settings from the manuscript table
------------------------------------------------
LSTM recurrent width : Integer [4, 64]
Dense width          : Integer [4, 64]
dropout p             : Real [0.00, 0.25]
L2 lambda             : Log-uniform [1e-7, 1e-3]
learning rate eta     : Log-uniform [2e-4, 3e-3]
BO calls              : 60
Initial evaluations   : 20
Trial training        : maximum 45 epochs
Final training        : maximum 80 epochs

Architecture
------------
LSTM -> Dense(ReLU) -> optional Dropout -> linear PM2.5 output

Install once
------------
pip install --upgrade pandas numpy scikit-learn scikit-optimize matplotlib tensorflow
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Replace the first "path" with the complete CSV path.
DATA_PATH = Path(r"path")

# Replace the second "path" with the directory in which outputs should be saved.
OUTPUT_ROOT = Path(r"path")

DATASET_NAME = "Dataset"

# Five independent BO + final-training runs used for mean ± standard deviation.
SEEDS = [42, 123, 2024, 7, 99]

# Set True only when TensorFlow must not use a GPU.
FORCE_CPU = False

CONFIG: dict[str, Any] = {
    # Forecast protocol.
    "window_steps": 8,
    "horizon_steps": 1,

    # Chronological target-time split.
    "train_fraction": 0.70,
    "validation_fraction": 0.15,

    # Causal missing-value handling.
    "predictor_ffill_limit_hours": 3,
    "minimum_sequences_each_split": 80,

    # Exact LSTM-BO table settings.
    "bo_calls": 60,
    "bo_initial_points": 20,
    "bo_trial_epochs": 45,
    "final_epochs": 80,

    # Training settings shared by BO trials and final training.
    "batch_size": 64,
    "early_stopping_patience": 12,
    "reduce_lr_patience": 6,
    "minimum_learning_rate": 1e-5,
    "verbose": 0,

    # Deployment evaluation.
    "evaluate_strict_int8": True,
    "representative_samples": 250,

    # Keep False for manuscript results. When True, only a short code test runs.
    "quick_mode": False,
    "quick_bo_calls": 4,
    "quick_bo_initial_points": 2,
    "quick_trial_epochs": 3,
    "quick_final_epochs": 5,
}

if FORCE_CPU:
    # TensorFlow has already been imported, so hide GPUs through its runtime API.
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

if CONFIG["quick_mode"]:
    CONFIG["bo_calls"] = int(CONFIG["quick_bo_calls"])
    CONFIG["bo_initial_points"] = min(
        int(CONFIG["quick_bo_initial_points"]),
        int(CONFIG["bo_calls"]) - 1,
    )
    CONFIG["bo_trial_epochs"] = int(CONFIG["quick_trial_epochs"])
    CONFIG["final_epochs"] = int(CONFIG["quick_final_epochs"])


# =============================================================================
# 2. REPRODUCIBILITY AND GENERAL UTILITIES
# =============================================================================


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def json_safe(value: Any) -> Any:
    """Convert NumPy, pathlib, and non-finite values to JSON-safe forms."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value if isinstance(value, str) else str(value)


def validate_paths() -> tuple[Path, Path]:
    data_path = DATA_PATH.expanduser()
    output_root = OUTPUT_ROOT.expanduser()

    if str(data_path).strip().lower() == "path":
        raise ValueError(
            'DATA_PATH is still "path". Replace it with the complete CSV path.'
        )
    if str(output_root).strip().lower() == "path":
        raise ValueError(
            'OUTPUT_ROOT is still "path". Replace it with an output directory.'
        )
    if not data_path.exists() or not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    return data_path.resolve(), output_root.resolve()


def write_json(path: Path, content: Any) -> None:
    path.write_text(
        json.dumps(json_safe(content), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# 3. GENERIC CSV LOADING AND COLUMN DETECTION
# =============================================================================


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z0-9.]+", " ", text).strip()


def find_timestamp_column(frame: pd.DataFrame) -> str:
    exact_candidates = [
        "Timestamp",
        "timestamp",
        "DateTime",
        "Datetime",
        "datetime",
        "Date Time",
        "date_time",
        "Date",
        "date",
    ]
    for candidate in exact_candidates:
        if candidate in frame.columns:
            return candidate

    normalized_aliases = {
        "timestamp",
        "datetime",
        "date time",
        "date",
        "time",
        "utc",
        "recorded at",
    }
    for column in frame.columns:
        if normalize_name(column) in normalized_aliases:
            return column

    raise ValueError(
        "Timestamp column was not found. Rename it to 'Timestamp' or add its "
        "name to find_timestamp_column()."
    )


def find_pm25_column(frame: pd.DataFrame) -> str:
    preferred = [
        "PM2.5 (µg/m³)",
        "PM2.5 (ug/m3)",
        "PM2.5",
        "PM25",
        "pm2.5",
        "pm25",
    ]
    for candidate in preferred:
        if candidate in frame.columns:
            return candidate

    for column in frame.columns:
        name = normalize_name(column).replace(" ", "")
        if re.search(r"pm2\.?5", name) or "pm25" in name:
            return column

    raise ValueError(
        "PM2.5 target column was not found. Rename it to 'PM2.5 (µg/m³)' "
        "or add its name to find_pm25_column()."
    )


def parse_datetime(series: pd.Series) -> pd.Series:
    """Try common formats and retain the parse with the highest success rate."""
    candidates: list[pd.Series] = []

    known_formats = [
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in known_formats:
        candidates.append(pd.to_datetime(series, format=fmt, errors="coerce"))

    candidates.append(pd.to_datetime(series, errors="coerce", dayfirst=False))
    candidates.append(pd.to_datetime(series, errors="coerce", dayfirst=True))

    parsed = max(candidates, key=lambda values: float(values.notna().mean()))
    success_rate = float(parsed.notna().mean())
    if success_rate < 0.80:
        raise ValueError(
            f"Only {success_rate:.1%} of timestamps could be parsed. "
            "Check the timestamp format."
        )
    return parsed


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace(r"^\s*(NA|N/A|None|null|nan)\s*$", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_hourly_data(path: Path) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """
    Read the CSV, detect timestamp/PM2.5 columns, aggregate duplicate hours,
    and restore an exact hourly grid.

    PM2.5 is never imputed. Predictor columns receive only causal forward fill.
    """
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError("The CSV contains no rows.")

    raw.columns = [str(column).strip() for column in raw.columns]
    timestamp_column = find_timestamp_column(raw)
    target_column = find_pm25_column(raw)

    timestamps = parse_datetime(raw[timestamp_column])
    valid_timestamp = timestamps.notna()
    raw = raw.loc[valid_timestamp].copy()
    raw.index = pd.DatetimeIndex(timestamps.loc[valid_timestamp])
    raw = raw.drop(columns=[timestamp_column]).sort_index()

    numeric = pd.DataFrame(index=raw.index)
    for column in raw.columns:
        values = clean_numeric(raw[column])
        if column == target_column or float(values.notna().mean()) >= 0.25:
            numeric[column] = values

    if target_column not in numeric.columns:
        numeric[target_column] = clean_numeric(raw[target_column])

    numeric = numeric.groupby(level=0).mean(numeric_only=True).sort_index()
    if numeric.empty:
        raise ValueError("No usable numeric observations remained after cleaning.")

    hourly_index = pd.date_range(
        start=numeric.index.min(),
        end=numeric.index.max(),
        freq="h",
    )
    hourly = numeric.reindex(hourly_index).replace([np.inf, -np.inf], np.nan)

    predictor_columns = [
        column for column in hourly.columns if column != target_column
    ]
    if predictor_columns:
        hourly[predictor_columns] = hourly[predictor_columns].ffill(
            limit=int(CONFIG["predictor_ffill_limit_hours"])
        )

    valid_targets = int(hourly[target_column].notna().sum())
    if valid_targets < 500:
        raise ValueError(
            f"Only {valid_targets} observed PM2.5 values remain; at least 500 are required."
        )

    audit = {
        "dataset_name": DATASET_NAME,
        "source_file": str(path),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "rows_after_timestamp_filter": int(len(raw)),
        "hourly_grid_rows": int(len(hourly)),
        "hourly_start": str(hourly.index.min()),
        "hourly_end": str(hourly.index.max()),
        "valid_pm25_values": valid_targets,
        "pm25_missing_after_hourly_reindex": int(hourly[target_column].isna().sum()),
        "predictor_columns": predictor_columns,
    }
    return hourly, target_column, audit


# =============================================================================
# 4. LEAKAGE-SAFE FEATURE PREPARATION
# =============================================================================


@dataclass
class PreparedData:
    feature_names: list[str]
    train_medians: pd.Series
    input_scaler: RobustScaler
    target_scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    y_train_raw: np.ndarray
    train_target_times: pd.DatetimeIndex
    X_val: np.ndarray
    y_val: np.ndarray
    y_val_raw: np.ndarray
    val_target_times: pd.DatetimeIndex
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    test_target_times: pd.DatetimeIndex


def build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=np.float32)
    weekday = index.dayofweek.to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "weekday_sin": np.sin(2.0 * np.pi * weekday / 7.0),
            "weekday_cos": np.cos(2.0 * np.pi * weekday / 7.0),
        },
        index=index,
    )


def prepare_data(hourly: pd.DataFrame, target_column: str) -> PreparedData:
    """
    Form 8-hour windows and assign each sequence to train/validation/test using
    the future target timestamp, preventing boundary leakage.
    """
    window = int(CONFIG["window_steps"])
    horizon = int(CONFIG["horizon_steps"])
    n_rows = len(hourly)

    train_end = int(n_rows * float(CONFIG["train_fraction"]))
    validation_end = int(
        n_rows
        * float(CONFIG["train_fraction"] + CONFIG["validation_fraction"])
    )

    if not (window < train_end < validation_end < n_rows):
        raise ValueError("The chronological split is invalid for this dataset size.")

    pm25_raw = hourly[target_column].to_numpy(dtype=np.float32)

    numeric_features = hourly.select_dtypes(include=[np.number]).copy()
    current_pm25_name = "PM2.5_current"
    while current_pm25_name in numeric_features.columns and current_pm25_name != target_column:
        current_pm25_name += "_input"
    numeric_features = numeric_features.rename(
        columns={target_column: current_pm25_name}
    )

    features = pd.concat(
        [numeric_features, build_time_features(hourly.index)],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    # Medians are computed from training rows only and then applied everywhere.
    train_medians = features.iloc[:train_end].median(axis=0).fillna(0.0)
    features = features.fillna(train_medians).fillna(0.0)

    input_scaler = RobustScaler()
    input_scaler.fit(
        features.iloc[:train_end].to_numpy(dtype=np.float32)
    )
    all_x = input_scaler.transform(
        features.to_numpy(dtype=np.float32)
    ).astype(np.float32)

    finite_training_targets = pm25_raw[:train_end][
        np.isfinite(pm25_raw[:train_end])
    ]
    if len(finite_training_targets) < 100:
        raise ValueError("Fewer than 100 finite PM2.5 training targets remain.")

    target_scaler = StandardScaler()
    target_scaler.fit(finite_training_targets.reshape(-1, 1))

    all_y_scaled = np.full(n_rows, np.nan, dtype=np.float32)
    observed_target = np.isfinite(pm25_raw)
    all_y_scaled[observed_target] = target_scaler.transform(
        pm25_raw[observed_target].reshape(-1, 1)
    ).ravel()

    packs: dict[str, dict[str, list[Any]]] = {
        split: {"X": [], "y": [], "raw": [], "time": []}
        for split in ("train", "val", "test")
    }

    first_anchor = window - 1
    last_anchor = n_rows - horizon - 1

    for anchor in range(first_anchor, last_anchor + 1):
        target_index = anchor + horizon
        start_index = anchor - window + 1
        x_window = all_x[start_index : anchor + 1]
        pm25_history = pm25_raw[start_index : anchor + 1]

        # Current/historical PM2.5 and the future target must be observed.
        if x_window.shape != (window, all_x.shape[1]):
            continue
        if not np.isfinite(pm25_history).all():
            continue
        if not np.isfinite(pm25_raw[target_index]):
            continue

        if target_index < train_end:
            split = "train"
        elif target_index < validation_end:
            split = "val"
        else:
            split = "test"

        packs[split]["X"].append(x_window)
        packs[split]["y"].append(all_y_scaled[target_index])
        packs[split]["raw"].append(pm25_raw[target_index])
        packs[split]["time"].append(hourly.index[target_index])

    arrays: dict[str, Any] = {}
    for split, values in packs.items():
        arrays[f"X_{split}"] = np.asarray(values["X"], dtype=np.float32)
        arrays[f"y_{split}"] = np.asarray(
            values["y"], dtype=np.float32
        ).reshape(-1, 1)
        arrays[f"y_{split}_raw"] = np.asarray(
            values["raw"], dtype=np.float32
        ).reshape(-1, 1)
        arrays[f"{split}_target_times"] = pd.DatetimeIndex(values["time"])

    counts = {
        split: len(arrays[f"X_{split}"])
        for split in ("train", "val", "test")
    }
    if min(counts.values()) < int(CONFIG["minimum_sequences_each_split"]):
        raise ValueError(
            "Too few valid sequences after causal filtering: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
        )

    return PreparedData(
        feature_names=list(features.columns),
        train_medians=train_medians,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        X_train=arrays["X_train"],
        y_train=arrays["y_train"],
        y_train_raw=arrays["y_train_raw"],
        train_target_times=arrays["train_target_times"],
        X_val=arrays["X_val"],
        y_val=arrays["y_val"],
        y_val_raw=arrays["y_val_raw"],
        val_target_times=arrays["val_target_times"],
        X_test=arrays["X_test"],
        y_test=arrays["y_test"],
        y_test_raw=arrays["y_test_raw"],
        test_target_times=arrays["test_target_times"],
    )


# =============================================================================
# 5. LSTM MODEL AND EXACT BAYESIAN SEARCH DOMAIN
# =============================================================================


SEARCH_SPACE = [
    Integer(4, 64, name="lstm_units"),
    Integer(4, 64, name="dense_units"),
    Real(0.0, 0.25, prior="uniform", name="dropout"),
    Real(1e-7, 1e-3, prior="log-uniform", name="l2"),
    Real(2e-4, 3e-3, prior="log-uniform", name="learning_rate"),
]


def values_to_spec(values: list[Any]) -> dict[str, Any]:
    return {
        "lstm_units": int(values[0]),
        "dense_units": int(values[1]),
        "dropout": float(values[2]),
        "l2": float(values[3]),
        "learning_rate": float(values[4]),
    }


def build_lstm(
    input_shape: tuple[int, int],
    specification: dict[str, Any],
) -> keras.Model:
    """
    Build the compact recurrent forecaster used for LSTM-BO.

    Architecture:
      LSTM -> Dense(ReLU) -> optional Dropout -> linear output

    ``unroll=True`` is suitable for the fixed eight-step input window and
    improves strict full-integer TFLite conversion compatibility on many
    TensorFlow versions. It does not expose future target observations.
    """
    regularizer = regularizers.l2(float(specification["l2"]))
    dropout = float(specification["dropout"])

    inputs = keras.Input(shape=input_shape, name="hourly_input")
    x = layers.LSTM(
        units=int(specification["lstm_units"]),
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        unroll=True,
        name="lstm",
    )(inputs)

    x = layers.Dense(
        int(specification["dense_units"]),
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense",
    )(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout, name="dropout")(x)

    outputs = layers.Dense(1, name="pm25_scaled")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_BO")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(specification["learning_rate"])
        ),
        loss="mse",
    )
    return model


def training_callbacks() -> list[keras.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(CONFIG["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=int(CONFIG["reduce_lr_patience"]),
            min_lr=float(CONFIG["minimum_learning_rate"]),
        ),
        TerminateOnNaN(),
    ]


def train_model(
    data: PreparedData,
    specification: dict[str, Any],
    seed: int,
    epochs: int,
) -> tuple[keras.Model, keras.callbacks.History]:
    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(seed)

    model = build_lstm(data.X_train.shape[1:], specification)
    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=int(epochs),
        batch_size=int(CONFIG["batch_size"]),
        callbacks=training_callbacks(),
        verbose=int(CONFIG["verbose"]),
        shuffle=False,
    )
    return model, history


def inverse_target(data: PreparedData, scaled_values: np.ndarray) -> np.ndarray:
    return data.target_scaler.inverse_transform(
        np.asarray(scaled_values, dtype=np.float32).reshape(-1, 1)
    ).ravel()


def calculate_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prediction = np.asarray(y_prediction, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(y_true, y_prediction))
    return {
        "R2": float(r2_score(y_true, y_prediction)),
        "MAE_ugm3": float(mean_absolute_error(y_true, y_prediction)),
        "RMSE_ugm3": float(np.sqrt(mse)),
        "MSE_ugm3_sq": mse,
    }


def optimize_lstm(
    data: PreparedData,
    seed: int,
    run_directory: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Run Gaussian-process Bayesian optimization.

    The objective is exactly 1 - validation R². There is no test-set term and
    no parameter-count penalty, so model selection is validation-R²-only.
    """
    trial_rows: list[dict[str, Any]] = []

    def objective(values: list[Any]) -> float:
        specification = values_to_spec(values)
        trial_number = len(trial_rows) + 1

        try:
            model, history = train_model(
                data=data,
                specification=specification,
                seed=seed,
                epochs=int(CONFIG["bo_trial_epochs"]),
            )
            validation_scaled = model.predict(data.X_val, verbose=0)
            validation_prediction = inverse_target(data, validation_scaled)
            validation_metrics = calculate_metrics(
                data.y_val_raw,
                validation_prediction,
            )
            validation_r2 = float(validation_metrics["R2"])
            objective_value = 1.0 - validation_r2

            trial_rows.append(
                {
                    "trial": trial_number,
                    "seed": seed,
                    "validation_R2": validation_r2,
                    "validation_MAE_ugm3": validation_metrics["MAE_ugm3"],
                    "validation_RMSE_ugm3": validation_metrics["RMSE_ugm3"],
                    "params": int(model.count_params()),
                    "epochs_completed": int(len(history.history.get("loss", []))),
                    "objective_1_minus_validation_R2": objective_value,
                    "status": "success",
                    "error": "",
                    **specification,
                }
            )

            del model, history
            tf.keras.backend.clear_session()
            gc.collect()
            return float(objective_value)

        except Exception as error:
            trial_rows.append(
                {
                    "trial": trial_number,
                    "seed": seed,
                    "validation_R2": np.nan,
                    "validation_MAE_ugm3": np.nan,
                    "validation_RMSE_ugm3": np.nan,
                    "params": np.nan,
                    "epochs_completed": 0,
                    "objective_1_minus_validation_R2": 1e6,
                    "status": "failed",
                    "error": repr(error),
                    **specification,
                }
            )
            tf.keras.backend.clear_session()
            gc.collect()
            return 1e6

    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=int(CONFIG["bo_calls"]),
        n_initial_points=int(CONFIG["bo_initial_points"]),
        random_state=seed,
        acq_func="EI",
    )

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(run_directory / "LSTM_BO_trials.csv", index=False)

    best_specification = values_to_spec(result.x)
    write_json(
        run_directory / "LSTM_BO_selected_specification.json",
        best_specification,
    )

    optimization_result = {
        "seed": seed,
        "best_objective": float(result.fun),
        "best_validation_R2_from_objective": float(1.0 - result.fun),
        "bo_calls": int(CONFIG["bo_calls"]),
        "bo_initial_points": int(CONFIG["bo_initial_points"]),
        "trial_maximum_epochs": int(CONFIG["bo_trial_epochs"]),
        "selected_specification": best_specification,
    }
    write_json(
        run_directory / "LSTM_BO_optimization_summary.json",
        optimization_result,
    )

    return best_specification, trials


# =============================================================================
# 6. STRICT FULL-INTEGER INT8 TFLITE CONVERSION
# =============================================================================


def representative_dataset(
    x_train: np.ndarray,
    maximum_samples: int,
) -> Iterator[list[np.ndarray]]:
    sample_count = min(int(maximum_samples), len(x_train))
    positions = np.linspace(
        0,
        len(x_train) - 1,
        num=sample_count,
        dtype=int,
    )
    for position in positions:
        yield [x_train[position : position + 1].astype(np.float32)]


def convert_to_strict_int8(
    model: keras.Model,
    x_train: np.ndarray,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        x_train,
        int(CONFIG["representative_samples"]),
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_blob = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=model_blob)
    interpreter.allocate_tensors()
    input_dtype = interpreter.get_input_details()[0]["dtype"]
    output_dtype = interpreter.get_output_details()[0]["dtype"]
    if input_dtype != np.int8 or output_dtype != np.int8:
        raise RuntimeError(
            "TFLite conversion did not produce INT8 input and output tensors."
        )
    return model_blob


def predict_strict_int8(
    model_blob: bytes,
    x_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    interpreter = tf.lite.Interpreter(model_content=model_blob)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0.0 or output_scale <= 0.0:
        raise RuntimeError("Invalid TFLite quantization scale.")

    predictions: list[np.ndarray] = []
    input_saturated = 0
    input_total = 0
    output_saturated = 0
    output_total = 0

    for sample in x_values:
        quantized = np.round(sample / input_scale + input_zero_point)
        input_saturated += int(
            np.sum((quantized < -128) | (quantized > 127))
        )
        input_total += int(quantized.size)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        interpreter.set_tensor(
            input_detail["index"],
            quantized[np.newaxis, ...],
        )
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(
            output_detail["index"]
        ).astype(np.int16)

        output_saturated += int(
            np.sum((quantized_output <= -128) | (quantized_output >= 127))
        )
        output_total += int(quantized_output.size)

        dequantized = (
            quantized_output.astype(np.float32) - float(output_zero_point)
        ) * float(output_scale)
        predictions.append(dequantized.ravel())

    diagnostics = {
        "input_saturation_rate": float(
            input_saturated / max(1, input_total)
        ),
        "output_saturation_rate": float(
            output_saturated / max(1, output_total)
        ),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }
    return np.concatenate(predictions), diagnostics


# =============================================================================
# 7. FIGURES, PREDICTIONS, AND SAVED PREPROCESSING OBJECTS
# =============================================================================


def save_bo_figures(trials: pd.DataFrame, run_directory: Path, seed: int) -> None:
    valid = trials.dropna(subset=["validation_R2"]).copy()
    if valid.empty:
        return

    valid["best_so_far_R2"] = valid["validation_R2"].cummax()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(valid["trial"], valid["validation_R2"], marker="o", label="Trial")
    plt.plot(valid["trial"], valid["best_so_far_R2"], label="Best so far")
    plt.xlabel("Bayesian-optimization trial")
    plt.ylabel("Validation $R^2$")
    plt.title(f"LSTM-BO validation convergence: seed {seed}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_directory / "LSTM_BO_validation_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.scatter(valid["params"], valid["validation_R2"])
    plt.xlabel("Trainable parameters")
    plt.ylabel("Validation $R^2$")
    plt.title(f"LSTM-BO validation accuracy versus parameters: seed {seed}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(run_directory / "LSTM_BO_accuracy_parameter_tradeoff.png", dpi=300)
    plt.close()


def save_preprocessing_objects(data: PreparedData, output_root: Path) -> None:
    with (output_root / "input_robust_scaler.pkl").open("wb") as file:
        pickle.dump(data.input_scaler, file)
    with (output_root / "target_standard_scaler.pkl").open("wb") as file:
        pickle.dump(data.target_scaler, file)
    with (output_root / "training_feature_medians.pkl").open("wb") as file:
        pickle.dump(data.train_medians, file)
    write_json(output_root / "feature_order.json", data.feature_names)


def make_prediction_frame(
    timestamps: pd.DatetimeIndex,
    observed: np.ndarray,
    predicted: np.ndarray,
    split: str,
    precision: str,
    seed: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "split": split,
            "precision": precision,
            "seed": seed,
            "observed_PM2.5_ugm3": np.asarray(observed).reshape(-1),
            "predicted_PM2.5_ugm3": np.asarray(predicted).reshape(-1),
        }
    )


# =============================================================================
# 8. ONE COMPLETE SEED RUN
# =============================================================================


def run_one_seed(
    data: PreparedData,
    seed: int,
    output_root: Path,
) -> pd.DataFrame:
    run_directory = output_root / f"seed_{seed}"
    run_directory.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"LSTM-BO | dataset={DATASET_NAME} | seed={seed}")
    print(f"BO calls={CONFIG['bo_calls']}, initial={CONFIG['bo_initial_points']}, "
          f"trial epochs={CONFIG['bo_trial_epochs']}, final epochs={CONFIG['final_epochs']}")
    print("=" * 100)

    best_specification, trials = optimize_lstm(
        data=data,
        seed=seed,
        run_directory=run_directory,
    )
    save_bo_figures(trials, run_directory, seed)

    final_model, final_history = train_model(
        data=data,
        specification=best_specification,
        seed=seed,
        epochs=int(CONFIG["final_epochs"]),
    )

    final_model.save(run_directory / "LSTM_BO_FP32.keras")
    pd.DataFrame(final_history.history).to_csv(
        run_directory / "LSTM_BO_final_training_history.csv",
        index=False,
    )

    validation_fp32 = inverse_target(
        data,
        final_model.predict(data.X_val, verbose=0),
    )
    test_fp32 = inverse_target(
        data,
        final_model.predict(data.X_test, verbose=0),
    )

    parameter_count = int(final_model.count_params())
    validation_fp32_metrics = calculate_metrics(
        data.y_val_raw,
        validation_fp32,
    )
    test_fp32_metrics = calculate_metrics(data.y_test_raw, test_fp32)

    result_rows: list[dict[str, Any]] = [
        {
            "dataset": DATASET_NAME,
            "model": "LSTM-BO",
            "teacher": "None",
            "seed": seed,
            "precision": "FP32",
            "validation_R2": validation_fp32_metrics["R2"],
            "validation_MAE_ugm3": validation_fp32_metrics["MAE_ugm3"],
            "validation_RMSE_ugm3": validation_fp32_metrics["RMSE_ugm3"],
            "R2": test_fp32_metrics["R2"],
            "MAE_ugm3": test_fp32_metrics["MAE_ugm3"],
            "RMSE_ugm3": test_fp32_metrics["RMSE_ugm3"],
            "MSE_ugm3_sq": test_fp32_metrics["MSE_ugm3_sq"],
            "params": parameter_count,
            "model_size_KB": float(
                (run_directory / "LSTM_BO_FP32.keras").stat().st_size / 1024.0
            ),
            "tflite_size_KB": np.nan,
            "quantization_R2_drop": 0.0,
            "input_saturation_rate": np.nan,
            "output_saturation_rate": np.nan,
            "epochs_completed": int(len(final_history.history.get("loss", []))),
            "status": "success",
            "specification": json.dumps(
                json_safe(best_specification), sort_keys=True
            ),
        }
    ]

    prediction_frames = [
        make_prediction_frame(
            data.val_target_times,
            data.y_val_raw,
            validation_fp32,
            "validation",
            "FP32",
            seed,
        ),
        make_prediction_frame(
            data.test_target_times,
            data.y_test_raw,
            test_fp32,
            "test",
            "FP32",
            seed,
        ),
    ]

    if bool(CONFIG["evaluate_strict_int8"]):
        try:
            int8_blob = convert_to_strict_int8(final_model, data.X_train)
            int8_path = run_directory / "LSTM_BO_strict_INT8.tflite"
            int8_path.write_bytes(int8_blob)

            validation_int8_scaled, validation_diagnostics = predict_strict_int8(
                int8_blob,
                data.X_val,
            )
            test_int8_scaled, test_diagnostics = predict_strict_int8(
                int8_blob,
                data.X_test,
            )
            validation_int8 = inverse_target(data, validation_int8_scaled)
            test_int8 = inverse_target(data, test_int8_scaled)

            validation_int8_metrics = calculate_metrics(
                data.y_val_raw,
                validation_int8,
            )
            test_int8_metrics = calculate_metrics(
                data.y_test_raw,
                test_int8,
            )

            result_rows.append(
                {
                    "dataset": DATASET_NAME,
                    "model": "LSTM-BO",
                    "teacher": "None",
                    "seed": seed,
                    "precision": "INT8",
                    "validation_R2": validation_int8_metrics["R2"],
                    "validation_MAE_ugm3": validation_int8_metrics["MAE_ugm3"],
                    "validation_RMSE_ugm3": validation_int8_metrics["RMSE_ugm3"],
                    "R2": test_int8_metrics["R2"],
                    "MAE_ugm3": test_int8_metrics["MAE_ugm3"],
                    "RMSE_ugm3": test_int8_metrics["RMSE_ugm3"],
                    "MSE_ugm3_sq": test_int8_metrics["MSE_ugm3_sq"],
                    "params": parameter_count,
                    "model_size_KB": float(int8_path.stat().st_size / 1024.0),
                    "tflite_size_KB": float(int8_path.stat().st_size / 1024.0),
                    "quantization_R2_drop": float(
                        test_fp32_metrics["R2"] - test_int8_metrics["R2"]
                    ),
                    "input_saturation_rate": test_diagnostics[
                        "input_saturation_rate"
                    ],
                    "output_saturation_rate": test_diagnostics[
                        "output_saturation_rate"
                    ],
                    "epochs_completed": int(
                        len(final_history.history.get("loss", []))
                    ),
                    "status": "strict_int8_success",
                    "specification": json.dumps(
                        json_safe(best_specification), sort_keys=True
                    ),
                    "validation_input_saturation_rate": validation_diagnostics[
                        "input_saturation_rate"
                    ],
                    "validation_output_saturation_rate": validation_diagnostics[
                        "output_saturation_rate"
                    ],
                }
            )

            prediction_frames.extend(
                [
                    make_prediction_frame(
                        data.val_target_times,
                        data.y_val_raw,
                        validation_int8,
                        "validation",
                        "INT8",
                        seed,
                    ),
                    make_prediction_frame(
                        data.test_target_times,
                        data.y_test_raw,
                        test_int8,
                        "test",
                        "INT8",
                        seed,
                    ),
                ]
            )

        except Exception as error:
            result_rows.append(
                {
                    "dataset": DATASET_NAME,
                    "model": "LSTM-BO",
                    "teacher": "None",
                    "seed": seed,
                    "precision": "INT8",
                    "validation_R2": np.nan,
                    "validation_MAE_ugm3": np.nan,
                    "validation_RMSE_ugm3": np.nan,
                    "R2": np.nan,
                    "MAE_ugm3": np.nan,
                    "RMSE_ugm3": np.nan,
                    "MSE_ugm3_sq": np.nan,
                    "params": parameter_count,
                    "model_size_KB": np.nan,
                    "tflite_size_KB": np.nan,
                    "quantization_R2_drop": np.nan,
                    "input_saturation_rate": np.nan,
                    "output_saturation_rate": np.nan,
                    "epochs_completed": int(
                        len(final_history.history.get("loss", []))
                    ),
                    "status": f"strict_int8_failed: {type(error).__name__}",
                    "error": repr(error),
                    "specification": json.dumps(
                        json_safe(best_specification), sort_keys=True
                    ),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        run_directory / "LSTM_BO_predictions.csv",
        index=False,
    )

    results = pd.DataFrame(result_rows)
    results.to_csv(run_directory / "LSTM_BO_results.csv", index=False)

    write_json(
        run_directory / "run_configuration.json",
        {
            "dataset": DATASET_NAME,
            "seed": seed,
            "config": CONFIG,
            "selected_specification": best_specification,
            "input_shape": list(data.X_train.shape[1:]),
            "number_of_features": len(data.feature_names),
            "train_sequences": len(data.X_train),
            "validation_sequences": len(data.X_val),
            "test_sequences": len(data.X_test),
        },
    )

    print("Selected specification:")
    print(json.dumps(json_safe(best_specification), indent=2))
    print("Final results:")
    print(results.round(6).to_string(index=False))

    del final_model, final_history
    tf.keras.backend.clear_session()
    gc.collect()
    return results


# =============================================================================
# 9. FIVE-SEED AGGREGATION
# =============================================================================


def aggregate_seed_results(all_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["dataset", "model", "teacher", "precision"]
    metric_columns = [
        "validation_R2",
        "validation_MAE_ugm3",
        "validation_RMSE_ugm3",
        "R2",
        "MAE_ugm3",
        "RMSE_ugm3",
        "MSE_ugm3_sq",
        "params",
        "model_size_KB",
        "tflite_size_KB",
        "quantization_R2_drop",
        "input_saturation_rate",
        "output_saturation_rate",
    ]
    metric_columns = [
        column for column in metric_columns if column in all_results.columns
    ]

    numeric_statistics = (
        all_results.groupby(group_columns, dropna=False)[metric_columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    numeric_statistics.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in numeric_statistics.columns
    ]

    formatted_rows: list[dict[str, Any]] = []
    for keys, group in all_results.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["number_of_seeds"] = int(group["seed"].nunique())

        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_mean_std"] = ""
                continue

            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = (
                f"{mean_value:.6f} ± {std_value:.6f}"
            )

        formatted_rows.append(row)

    formatted_summary = pd.DataFrame(formatted_rows)
    return numeric_statistics, formatted_summary


# =============================================================================
# 10. MAIN DRIVER
# =============================================================================


def main() -> None:
    data_path, output_root = validate_paths()
    experiment_directory = output_root / "LSTM_BO_outputs"
    experiment_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("GENERIC LSTM-BO PM2.5 FORECASTING")
    print("=" * 100)
    print(f"Dataset name : {DATASET_NAME}")
    print(f"Dataset path : {data_path}")
    print(f"Output path  : {experiment_directory}")
    print("Forecast     : 8 input hours -> immediate next hour PM2.5")
    print("Split        : chronological 70/15/15")
    print("BO selection : validation R² only")
    print(f"Seeds        : {SEEDS}")
    print("=" * 100)

    hourly, target_column, audit = load_hourly_data(data_path)
    prepared = prepare_data(hourly, target_column)

    pd.DataFrame([audit]).to_csv(
        experiment_directory / "dataset_schema_audit.csv",
        index=False,
    )
    save_preprocessing_objects(prepared, experiment_directory)
    write_json(
        experiment_directory / "experiment_configuration.json",
        {
            "dataset_name": DATASET_NAME,
            "data_path": data_path,
            "output_root": output_root,
            "seeds": SEEDS,
            "config": CONFIG,
            "search_domain": {
                "lstm_units": "Integer[4,64]",
                "dense_units": "Integer[4,64]",
                "dropout": "Real[0,0.25]",
                "l2": "LogUniform[1e-7,1e-3]",
                "learning_rate": "LogUniform[2e-4,3e-3]",
            },
            "feature_names": prepared.feature_names,
        },
    )

    print(f"Hourly rows         : {len(hourly):,}")
    print(f"Input features      : {len(prepared.feature_names)}")
    print(f"Training sequences  : {len(prepared.X_train):,}")
    print(f"Validation sequences: {len(prepared.X_val):,}")
    print(f"Test sequences      : {len(prepared.X_test):,}")

    all_seed_results: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for seed in SEEDS:
        try:
            result = run_one_seed(
                data=prepared,
                seed=int(seed),
                output_root=experiment_directory,
            )
            all_seed_results.append(result)
        except Exception as error:
            failures.append(
                {
                    "seed": int(seed),
                    "error_type": type(error).__name__,
                    "error": repr(error),
                }
            )
            print(f"Seed {seed} failed: {type(error).__name__}: {error}")
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    if failures:
        pd.DataFrame(failures).to_csv(
            experiment_directory / "LSTM_BO_failures.csv",
            index=False,
        )

    if not all_seed_results:
        raise RuntimeError(
            "No seed completed successfully. Check LSTM_BO_failures.csv."
        )

    combined_results = pd.concat(all_seed_results, ignore_index=True)
    combined_results.to_csv(
        experiment_directory / "LSTM_BO_all_seed_results.csv",
        index=False,
    )

    statistics, formatted_summary = aggregate_seed_results(combined_results)
    statistics.to_csv(
        experiment_directory / "LSTM_BO_five_seed_statistics.csv",
        index=False,
    )
    formatted_summary.to_csv(
        experiment_directory / "LSTM_BO_five_seed_mean_std.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("FIVE-SEED LSTM-BO SUMMARY")
    print("=" * 100)
    print(formatted_summary.to_string(index=False))
    print(f"\nAll outputs saved in: {experiment_directory}")


if __name__ == "__main__":
    main()