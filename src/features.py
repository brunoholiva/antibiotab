"""Molecular featurization: Morgan fingerprints + RDKit descriptors."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.base import BaseEstimator, TransformerMixin


def _compute_rdkit_descriptors_batch(smiles_batch: list[str]) -> np.ndarray:
    """Compute 200 RDKit descriptors for a batch of SMILES.

    Creates one ``RDKit2DNormalized`` instance per process to avoid pickling
    issues with joblib.  Molecules that fail or return a bad status get a
    zero vector.
    """
    from descriptastorus.descriptors import rdNormalizedDescriptors

    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    out = np.empty((len(smiles_batch), 200), dtype=np.float32)
    for i, smi in enumerate(smiles_batch):
        try:
            desc = generator.process(smi)
            out[i] = desc[1:] if desc[0] else 0.0
        except Exception:
            out[i] = 0.0
    return out


_MORGAN_GEN_CACHE: dict[tuple[int, int], object] = {}


def _get_morgan_gen(radius: int, fp_size: int) -> object:
    """Return a cached MorganGenerator for the given parameters."""
    key = (radius, fp_size)
    if key not in _MORGAN_GEN_CACHE:
        _MORGAN_GEN_CACHE[key] = GetMorganGenerator(radius=radius, fpSize=fp_size)
    return _MORGAN_GEN_CACHE[key]


def _chunk_list(lst: list, chunks: int) -> list[list]:
    """Split a list into roughly equal parts."""
    k, m = divmod(len(lst), chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(chunks)]


class MoleculeFeaturizer(BaseEstimator, TransformerMixin):
    """Convert SMILES to a concatenated feature matrix.

    The matrix is built by concatenating a **Morgan fingerprint** (512-bit,
    radius 2) with 200 **RDKit normalised descriptors** (computed via
    ``descriptastorus``).

    Parameters
    ----------
    n_bits : int, default=512
        Number of bits for the Morgan fingerprint.
    radius : int, default=2
        Morgan fingerprint radius.
    n_jobs : int, default=-1
        Number of parallel workers for the slow RDKit descriptor computation.
        ``-1`` uses all available CPUs.
    """

    def __init__(self, n_bits: int = 512, radius: int = 2, n_jobs: int = -1):
        self.n_bits = n_bits
        self.radius = radius
        self.n_jobs = n_jobs

    def fit(self, X: list[str], y: np.ndarray | None = None):
        """No-op; included for scikit-learn pipeline compatibility."""
        return self

    def _morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """Compute a single Morgan fingerprint bit-vector."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        gen = _get_morgan_gen(self.radius, self.n_bits)
        return np.array(gen.GetFingerprint(mol), dtype=np.float32)

    def _compute_morgan_fingerprints(self, smiles_list: list[str]) -> np.ndarray:
        """Morgan fingerprints with progress bar."""
        n = len(smiles_list)
        columns = [
            TextColumn("  "),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="white", complete_style="magenta"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ]
        with Progress(*columns) as progress:
            task = progress.add_task("[cyan]Morgan fingerprints", total=n)
            fps = np.empty((n, self.n_bits), dtype=np.float32)
            for i, smi in enumerate(smiles_list):
                fps[i] = self._morgan_fingerprint(smi)
                progress.advance(task)
        return fps

    def _compute_rdkit_descriptors(self, smiles_list: list[str]) -> np.ndarray:
        """RDKit descriptors computed in parallel across chunks."""
        n = len(smiles_list)
        njobs = effective_n_jobs(self.n_jobs)
        batches = _chunk_list(smiles_list, njobs * 2)
        columns = [
            TextColumn("  "),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="white", complete_style="magenta"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ]
        with Progress(*columns) as progress:
            task = progress.add_task("[green]RDKit descriptors", total=n)
            results = Parallel(n_jobs=njobs, return_as="generator")(
                delayed(_compute_rdkit_descriptors_batch)(batch) for batch in batches
            )
            parts = []
            for r in results:
                parts.append(r)
                progress.advance(task, advance=len(r))
            return np.vstack(parts)

    def transform(self, X: list[str], y: np.ndarray | None = None) -> np.ndarray:
        """Featurise a list of SMILES into a ``(n, 712)`` feature matrix.

        Parameters
        ----------
        X : list of str
            SMILES strings.

        Returns
        -------
        np.ndarray of shape ``(len(X), 712)``
            Concatenated Morgan (512) + RDKit (200) features.
        """
        morgan_fps = self._compute_morgan_fingerprints(X)
        rdkit_desc = self._compute_rdkit_descriptors(X)
        feats = np.hstack([morgan_fps, rdkit_desc])
        # Zero out the RDKit portion for molecules RDKit couldn't parse
        invalid = np.all(morgan_fps == 0, axis=1)
        feats[invalid, self.n_bits:] = 0.0
        return feats
