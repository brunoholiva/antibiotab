"""Entry-point: train a TabPFN classifier on GNEtolC data and save it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from tabpfn import TabPFNClassifier

from src.features import MoleculeFeaturizer
from src.model_utils import move_model_to_device

DATA_DIR = Path("data/model")
CSV_PATH = DATA_DIR / "GNEtolC.csv"
MODEL_PATH = DATA_DIR / "tabpfn_model.joblib"


def load_data(path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and binary targets from a CSV file."""
    df = pd.read_csv(path)
    return df["SMILES"].tolist(), df["target"].values


def train_model(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 8, softmax_temperature: float = 2.0
) -> TabPFNClassifier:
    """Fit a TabPFN classifier and return it.

    Uses ``fit_mode="fit_with_cache"`` to cache the transformer KV-cache
    after fitting, avoiding re-encoding the training set on each ``predict``
    call.
    """
    model = TabPFNClassifier(
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        device="cuda",
        fit_mode="fit_with_cache",
        show_progress_bar=True,
    )
    model.fit(X, y)
    return model


def main(
    train_csv: Path = CSV_PATH,
    model_out: Path = MODEL_PATH,
    n_jobs: int = 5,
) -> None:
    """Train a TabPFN model on *train_csv* and save to *model_out*."""
    print(f"Loading {train_csv} ...")
    smiles, y = load_data(train_csv)
    n_pos = int(y.sum())
    print(f"Loaded {len(smiles)} molecules  ({n_pos} positive, {len(y) - n_pos} negative)")

    featurizer = MoleculeFeaturizer(n_jobs=n_jobs)
    print("Computing features ...")
    X = featurizer.transform(smiles)
    print(f"Feature matrix shape: {X.shape}")

    print("Training TabPFNClassifier (n_estimators=8, softmax_temperature=2, "
          "device=cuda, fit_mode=fit_with_cache) ...")
    model = train_model(X, y)

    print(f"Saving model to {model_out} ...")
    move_model_to_device(model, "cpu")
    dump(model, model_out)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TabPFN classifier.")
    parser.add_argument("--train-csv", type=Path, default=CSV_PATH,
                        help="Path to training CSV (SMILES, target columns)")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH,
                        help="Path to save the trained model")
    parser.add_argument("--n-jobs", type=int, default=5,
                        help="Parallel workers for featurization")
    args = parser.parse_args()
    main(train_csv=args.train_csv, model_out=args.model_out, n_jobs=args.n_jobs)
