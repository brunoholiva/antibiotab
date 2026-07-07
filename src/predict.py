"""Predict using a trained TabPFN model."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from joblib import load
from tabpfn import TabPFNClassifier

from src.features import MoleculeFeaturizer
from src.file_io import split_zinc_file, stream_zinc
from src.model_utils import move_model_to_device


def _ensure_cuda_compat() -> None:
    """Set PyTorch CUDA allocator config for large model loading."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        extra = "expandable_segments:True"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{conf},{extra}" if conf else extra


def load_model(path: Path, device: str = "cpu") -> TabPFNClassifier:
    """Load a saved TabPFN classifier and optionally move it to *device*.

    Parameters
    ----------
    path : Path
        Path to the joblib-saved model.
    device : str, default="cpu"
        Target device (``"cpu"`` or ``"cuda"``).
    """
    model: TabPFNClassifier = load(path)
    model.inference_precision = "autocast"
    if device != "cpu":
        _ensure_cuda_compat()
        move_model_to_device(model, device)
    return model


def predict_from_features(
    X: np.ndarray, model: TabPFNClassifier
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities from a pre-featurized matrix.

    Parameters
    ----------
    X : np.ndarray of shape ``(n, n_features)``
        Feature matrix.
    model : TabPFNClassifier
        Fitted classifier.

    Returns
    -------
    preds : np.ndarray of shape ``(n,)``
        Predicted class labels (0 or 1).
    probs : np.ndarray of shape ``(n, 2)``
        Predicted class probabilities.
    """
    probs = model.predict_proba(X)
    preds = model.classes_[probs.argmax(axis=1)]
    return preds, probs


def predict(
    smiles: list[str], model: TabPFNClassifier, featurizer: MoleculeFeaturizer
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities for a list of SMILES.

    Parameters
    ----------
    smiles : list of str
        SMILES strings.
    model : TabPFNClassifier
        Fitted classifier.
    featurizer : MoleculeFeaturizer
        Featurizer with the same configuration used during training.

    Returns
    -------
    preds : np.ndarray of shape ``(n,)``
        Predicted class labels (0 or 1).
    probs : np.ndarray of shape ``(n, 2)``
        Predicted class probabilities.
    """
    X = featurizer.transform(smiles)
    return predict_from_features(X, model)


def predict_file(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int = 50000,
    device: str = "cuda",
    n_jobs: int = -1,
) -> None:
    """Score every SMILES in a file and write results to a CSV.

    Reads the input in batches, featurizes + predicts each batch, and
    appends results to *output_path*.  The output has four columns:
    ``zinc_id``, ``smiles``, ``prediction``, ``probability``.
    """
    model = load_model(model_path, device=device)
    featurizer = MoleculeFeaturizer(n_jobs=n_jobs)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zinc_id", "smiles", "prediction", "probability"])

        for smiles_batch, zinc_ids in stream_zinc(input_path, batch_size):
            if not smiles_batch:
                continue
            preds, probs = predict(smiles_batch, model, featurizer)
            for zid, smi, p, prob in zip(zinc_ids, smiles_batch, preds, probs):
                writer.writerow([zid, smi, p, f"{prob[1]:.6f}"])

    print(f"Results written to {output_path}")


def _run_split_zinc(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="split-zinc")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parsed = parser.parse_args(args)
    paths = split_zinc_file(parsed.input, parsed.output_dir, parsed.max_rows)
    print(f"Split {parsed.input.name} into {len(paths)} shards in {parsed.output_dir}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "split-zinc":
        _run_split_zinc(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Score molecules with TabPFN.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Tab-separated file with a 'smiles' column (not needed with --predict-dir)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="data/model/tabpfn_model.joblib",
    )
    parser.add_argument("--output", "-o", type=Path, default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cpu or cuda)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers for featurization")
    parser.add_argument("--featurize-dir", type=Path, default=None,
                        help="Featurize only, save features to DIR (no model/GPU)")
    parser.add_argument("--predict-dir", type=Path, default=None,
                        help="Predict from pre-computed features in DIR (no input file)")
    parser.add_argument("--features-f16", action="store_true",
                        help="Save features as float16 (half storage)")
    args = parser.parse_args()

    if args.featurize_dir:
        if args.input is None:
            parser.error("--featurize-dir requires a positional input file")
        from src.file_io import featurize_file
        featurize_file(
            input_path=args.input,
            features_dir=args.featurize_dir,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            features_f16=args.features_f16,
        )
    elif args.predict_dir:
        from src.file_io import predict_from_features_dir
        predict_from_features_dir(
            features_dir=args.predict_dir,
            output_path=args.output,
            model_path=args.model,
            device=args.device,
        )
    else:
        if args.input is None:
            parser.error("input file required (use --featurize-dir or --predict-dir "
                         "for split mode)")
        predict_file(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            batch_size=args.batch_size,
            device=args.device,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
