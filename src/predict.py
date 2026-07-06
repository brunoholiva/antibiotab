"""Predict using a trained TabPFN model."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from tabpfn import TabPFNClassifier

from src.features import MoleculeFeaturizer
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


def stream_smiles(path: Path, batch_size: int):
    """Yield batches of SMILES from a tab-separated file with a ``smiles`` column."""
    for chunk in pd.read_csv(path, sep="\t", chunksize=batch_size):
        yield chunk["smiles"].tolist()


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
    appends results to *output_path*.  The output has three columns:
    ``smiles``, ``prediction``, ``probability``.
    """
    model = load_model(model_path, device=device)
    featurizer = MoleculeFeaturizer(n_jobs=n_jobs)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "prediction", "probability"])

        for batch in stream_smiles(input_path, batch_size):
            if not batch:
                continue
            preds, probs = predict(batch, model, featurizer)
            for smi, p, prob in zip(batch, preds, probs):
                writer.writerow([smi, p, f"{prob[1]:.6f}"])

    print(f"Results written to {output_path}")


def predict_file_async(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int = 50000,
    device: str = "cuda",
    n_jobs: int = -1,
) -> None:
    """Score every SMILES overlapping CPU featurization with GPU prediction.

    Uses a background thread to featurize batch N+1 on CPU while batch N
    is being predicted on GPU.  Same output format as :func:`predict_file`.

    Parameters
    ----------
    input_path : Path
        Tab-separated file with a ``smiles`` column.
    output_path : Path
        Path to write the CSV output.
    model_path : Path
        Path to the saved TabPFN model.
    batch_size : int, default=50000
        Number of SMILES per batch.
    device : str, default="cuda"
        Device for inference (``"cpu"`` or ``"cuda"``).
    n_jobs : int, default=-1
        Parallel workers for featurization.
    """
    model = load_model(model_path, device=device)
    featurizer = MoleculeFeaturizer(n_jobs=n_jobs)

    stream = stream_smiles(input_path, batch_size)

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        open(output_path, "w", newline="") as f,
    ):
        writer = csv.writer(f)
        writer.writerow(["smiles", "prediction", "probability"])

        batch = next(stream, None)
        if not batch:
            return

        X_future = executor.submit(featurizer.transform, batch)

        while True:
            next_batch = next(stream, None)
            next_future = (
                executor.submit(featurizer.transform, next_batch)
                if next_batch
                else None
            )

            X = X_future.result()
            preds, probs = predict_from_features(X, model)
            for smi, p, prob in zip(batch, preds, probs):
                writer.writerow([smi, p, f"{prob[1]:.6f}"])

            if next_batch is None:
                break

            X_future = next_future
            batch = next_batch

    print(f"Results written to {output_path}")


def featurize_file(
    input_path: Path,
    features_dir: Path,
    batch_size: int = 50000,
    n_jobs: int = -1,
    features_f16: bool = False,
) -> None:
    """Featurize SMILES from a file and save features to *features_dir*.

    Reads *input_path* in batches, featurizes each batch via
    :class:`MoleculeFeaturizer`, and writes:

    - ``batch_NNNNNN.npy`` — feature matrix (float32 or float16)
    - ``batch_NNNNNN.smiles.csv`` — one SMILES per line (no header)
    - ``manifest.json`` — metadata (n_bits, radius, n_features, f16, batches)

    Does **not** require a model or GPU; only RDKit + joblib.

    Parameters
    ----------
    input_path : Path
        Tab-separated file with a ``smiles`` column.
    features_dir : Path
        Output directory (created if it does not exist).
    batch_size : int, default=50000
        Number of SMILES per batch / ``.npy`` file.
    n_jobs : int, default=-1
        Parallel workers for featurization.
    features_f16 : bool, default=False
        If True, save features as ``float16`` instead of ``float32``.
    """
    featurizer = MoleculeFeaturizer(n_jobs=n_jobs)
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "n_bits": featurizer.n_bits,
        "radius": featurizer.radius,
        "n_features": featurizer.n_features,
        "f16": features_f16,
        "batches": [],
    }

    batch_index = 0
    for smiles_batch in stream_smiles(input_path, batch_size):
        if not smiles_batch:
            continue

        X = featurizer.transform(smiles_batch)
        if features_f16:
            X = X.astype(np.float16)

        npy_path = features_dir / f"batch_{batch_index:06d}.npy"
        smi_path = features_dir / f"batch_{batch_index:06d}.smiles.csv"

        np.save(npy_path, X)
        smi_path.write_text("\n".join(smiles_batch) + "\n")

        manifest["batches"].append({"index": batch_index, "n_mols": len(smiles_batch)})
        batch_index += 1

    (features_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_total = sum(b["n_mols"] for b in manifest["batches"])
    print(f"Featurized {n_total} molecules into {features_dir} "
          f"({batch_index} batches)")


def predict_from_features_dir(
    features_dir: Path,
    output_path: Path,
    model_path: Path,
    device: str = "cuda",
) -> None:
    """Predict from pre-computed features saved by :func:`featurize_file`.

    Recursively scans *features_dir* for ``manifest.json`` files, loads
    each batch's ``.npy`` in global order (sorted by parent-directory path
    then batch index), predicts, and writes the result CSV.

    Parameters
    ----------
    features_dir : Path
        Directory containing sub-directories with ``manifest.json``,
        ``batch_*.npy``, and ``batch_*.smiles.csv``.
    output_path : Path
        CSV output path.
    model_path : Path
        Path to the saved TabPFN model.
    device : str, default="cuda"
        Device for inference (``"cpu"`` or ``"cuda"``).
    """
    model = load_model(model_path, device=device)

    manifest_paths = sorted(features_dir.rglob("manifest.json"))
    if not manifest_paths:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "prediction", "probability"])
        print(f"No features found in {features_dir}")
        return

    batches: list[tuple[Path, int]] = []
    for mp in manifest_paths:
        parent = mp.parent
        manifest = json.loads(mp.read_text())
        for batch_info in manifest["batches"]:
            batches.append((parent, batch_info["index"]))
    batches.sort(key=lambda x: (str(x[0]), x[1]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "prediction", "probability"])

        for parent_dir, batch_idx in batches:
            npy_path = parent_dir / f"batch_{batch_idx:06d}.npy"
            smi_path = parent_dir / f"batch_{batch_idx:06d}.smiles.csv"

            if not npy_path.exists() or not smi_path.exists():
                print(f"Warning: missing {npy_path} or {smi_path}, skipping")
                continue

            X = np.load(npy_path)
            if X.dtype == np.float16:
                X = X.astype(np.float32)

            smiles = [s for s in smi_path.read_text().strip().split("\n") if s]
            if not smiles:
                continue

            preds, probs = predict_from_features(X, model)
            for smi, p, prob in zip(smiles, preds, probs):
                writer.writerow([smi, p, f"{prob[1]:.6f}"])

    n_total = sum(1 for _ in open(output_path)) - 1
    print(f"Predicted {n_total} molecules, results written to {output_path}")


def main() -> None:
    import argparse

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
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Overlap CPU featurization with GPU prediction")
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
        featurize_file(
            input_path=args.input,
            features_dir=args.featurize_dir,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            features_f16=args.features_f16,
        )
    elif args.predict_dir:
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
        runner = predict_file_async if args.use_async else predict_file
        runner(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            batch_size=args.batch_size,
            device=args.device,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
