"""Entry-point: train a TabPFN classifier on GNEtolC data and save it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from tabpfn import TabPFNClassifier

from src.features import MoleculeFeaturizer

DATA_DIR = Path("data/model")
CSV_PATH = DATA_DIR / "GNEtolC.csv"
MODEL_PATH = DATA_DIR / "tabpfn_model.joblib"


def load_data(path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and binary targets from a CSV file."""
    df = pd.read_csv(path)
    return df["SMILES"].tolist(), df["target"].values


def train_model(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 32, softmax_temperature: float = 2.0
) -> TabPFNClassifier:
    """Fit a TabPFN classifier and return it."""
    model = TabPFNClassifier(
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        show_progress_bar=True,
    )
    model.fit(X, y)
    return model


def main() -> None:
    print(f"Loading {CSV_PATH} ...")
    smiles, y = load_data(CSV_PATH)
    n_pos = int(y.sum())
    print(f"Loaded {len(smiles)} molecules  ({n_pos} positive, {len(y) - n_pos} negative)")

    featurizer = MoleculeFeaturizer(n_jobs=-1)
    print("Computing features ...")
    X = featurizer.transform(smiles)
    print(f"Feature matrix shape: {X.shape}")

    print("Training TabPFNClassifier (n_estimators=32, softmax_temperature=2) ...")
    model = train_model(X, y)

    print(f"Saving model to {MODEL_PATH} ...")
    dump(model, MODEL_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
