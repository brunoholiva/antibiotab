"""File I/O operations: streaming, featurization, splitting, and prediction from disk."""

from __future__ import annotations

import csv
import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import MoleculeFeaturizer


def stream_zinc(
    path: Path, batch_size: int
) -> Generator[tuple[list[str], list[str]], None, None]:
    """Yield ``(smiles, zinc_ids)`` batches from a TSV with ``smiles`` and ``zinc_id`` columns.

    Parameters
    ----------
    path : Path
        Tab-separated file.
    batch_size : int
        Number of rows per batch.

    Yields
    ------
    tuple of (list of str, list of str)
        ``(smiles_batch, zinc_ids_batch)``.
    """
    for chunk in pd.read_csv(path, sep="\t", chunksize=batch_size):
        yield chunk["smiles"].tolist(), chunk["zinc_id"].tolist()


def split_zinc_file(
    input_path: Path,
    output_dir: Path,
    max_rows: int = 2_000_000,
) -> list[Path]:
    """Split a ZINC TSV file into smaller shards, each with a header line.

    Shard files are named ``{stem}_shard_{index:04d}.txt`` in *output_dir*.
    The header row is preserved in every shard.

    Parameters
    ----------
    input_path : Path
        Input TSV file with a header row.
    output_dir : Path
        Directory for the shard files (created if it does not exist).
    max_rows : int, default=2_000_000
        Maximum number of data rows per shard.

    Returns
    -------
    list of Path
        Paths to the created shard files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    shard_paths: list[Path] = []

    with open(input_path) as f:
        header = f.readline()
        if not header:
            return shard_paths

        shard_index = 0
        while True:
            lines = [header]
            for _ in range(max_rows):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            if len(lines) <= 1:
                break

            shard_path = output_dir / f"{stem}_shard_{shard_index:04d}.txt"
            shard_path.write_text("".join(lines))
            shard_paths.append(shard_path)
            shard_index += 1

    return shard_paths


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
    - ``batch_NNNNNN.rows.csv`` — ``zinc_id,smiles`` for each molecule
    - ``manifest.json`` — metadata (n_bits, radius, n_features, f16, batches)

    Does **not** require a model or GPU; only RDKit + joblib.

    Parameters
    ----------
    input_path : Path
        Tab-separated file with ``smiles`` and ``zinc_id`` columns.
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
    for smiles_batch, zinc_ids in stream_zinc(input_path, batch_size):
        if not smiles_batch:
            continue

        X = featurizer.transform(smiles_batch)
        if features_f16:
            X = X.astype(np.float16)

        npy_path = features_dir / f"batch_{batch_index:06d}.npy"
        rows_path = features_dir / f"batch_{batch_index:06d}.rows.csv"

        np.save(npy_path, X)
        df_rows = pd.DataFrame({"zinc_id": zinc_ids, "smiles": smiles_batch})
        df_rows.to_csv(rows_path, index=False)

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
        ``batch_*.npy``, and ``batch_*.rows.csv``.
    output_path : Path
        CSV output path.
    model_path : Path
        Path to the saved TabPFN model.
    device : str, default="cuda"
        Device for inference (``"cpu"`` or ``"cuda"``).
    """
    from src.predict import load_model, predict_from_features

    model = load_model(model_path, device=device)

    manifest_paths = sorted(features_dir.rglob("manifest.json"))
    if not manifest_paths:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zinc_id", "smiles", "prediction", "probability"])
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
        writer.writerow(["zinc_id", "smiles", "prediction", "probability"])

        for parent_dir, batch_idx in batches:
            npy_path = parent_dir / f"batch_{batch_idx:06d}.npy"
            rows_path = parent_dir / f"batch_{batch_idx:06d}.rows.csv"

            if not npy_path.exists() or not rows_path.exists():
                print(f"Warning: missing {npy_path} or {rows_path}, skipping")
                continue

            X = np.load(npy_path)
            if X.dtype == np.float16:
                X = X.astype(np.float32)

            rows = pd.read_csv(rows_path)
            if rows.empty:
                continue

            preds, probs = predict_from_features(X, model)
            for zid, smi, p, prob in zip(
                rows["zinc_id"], rows["smiles"], preds, probs
            ):
                writer.writerow([zid, smi, int(p), f"{prob[1]:.6f}"])

    n_total = sum(1 for _ in open(output_path)) - 1
    print(f"Predicted {n_total} molecules, results written to {output_path}")
