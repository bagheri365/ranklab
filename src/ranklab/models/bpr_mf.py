"""Deterministic NumPy BPR matrix factorization core.

This module is intentionally separate from the scaffold's public ``bpr.py``
entry point. M0.8a freezes and tests the numerical core first; M0.8b will wire
it into the repository's config/CLI surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import numpy as np


Id = Hashable


@dataclass(frozen=True)
class IdIndex:
    """Stable mapping between raw ids and contiguous integer indices."""

    values: tuple[Id, ...]

    @classmethod
    def from_values(cls, values: Iterable[Id]) -> "IdIndex":
        unique = {value for value in values}
        return cls(tuple(sorted(unique, key=repr)))

    @property
    def to_index(self) -> dict[Id, int]:
        return {value: idx for idx, value in enumerate(self.values)}

    def encode(self, value: Id) -> int:
        try:
            return self.to_index[value]
        except KeyError as exc:
            raise KeyError(f"unknown id: {value!r}") from exc

    def decode(self, index: int) -> Id:
        return self.values[index]


@dataclass
class BPRMatrixFactorization:
    """Matrix-factorization scorer trained with a BPR pairwise objective."""

    user_index: IdIndex
    item_index: IdIndex
    user_factors: np.ndarray
    item_factors: np.ndarray

    @classmethod
    def initialize(
        cls,
        user_ids: Iterable[Id],
        item_ids: Iterable[Id],
        *,
        embedding_dim: int,
        seed: int,
        init_std: float = 0.01,
    ) -> "BPRMatrixFactorization":
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        user_index = IdIndex.from_values(user_ids)
        item_index = IdIndex.from_values(item_ids)
        if not user_index.values:
            raise ValueError("at least one user is required")
        if not item_index.values:
            raise ValueError("at least one item is required")

        rng = np.random.default_rng(seed)
        user_factors = rng.normal(
            0.0, init_std, size=(len(user_index.values), embedding_dim)
        ).astype(np.float64)
        item_factors = rng.normal(
            0.0, init_std, size=(len(item_index.values), embedding_dim)
        ).astype(np.float64)
        return cls(user_index, item_index, user_factors, item_factors)

    @property
    def embedding_dim(self) -> int:
        return int(self.user_factors.shape[1])

    def score(self, user_id: Id, item_id: Id) -> float:
        u = self.user_index.encode(user_id)
        i = self.item_index.encode(item_id)
        return float(np.dot(self.user_factors[u], self.item_factors[i]))

    def score_items(self, user_id: Id, item_ids: Sequence[Id]) -> np.ndarray:
        u = self.user_index.encode(user_id)
        item_map = self.item_index.to_index
        indices = np.fromiter(
            (item_map[item_id] for item_id in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        return self.item_factors[indices] @ self.user_factors[u]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "ranklab_bpr_npz_v1",
            "embedding_dim": self.embedding_dim,
            "user_ids": list(self.user_index.values),
            "item_ids": list(self.item_index.values),
        }
        np.savez_compressed(
            destination,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            metadata=np.array(json.dumps(metadata, separators=(",", ":"))),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPRMatrixFactorization":
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            if metadata.get("format") != "ranklab_bpr_npz_v1":
                raise ValueError("unsupported BPR checkpoint format")
            user_factors = np.asarray(checkpoint["user_factors"], dtype=np.float64)
            item_factors = np.asarray(checkpoint["item_factors"], dtype=np.float64)

        expected_dim = int(metadata["embedding_dim"])
        if user_factors.ndim != 2 or item_factors.ndim != 2:
            raise ValueError("invalid BPR checkpoint factor shape")
        if user_factors.shape[1] != expected_dim or item_factors.shape[1] != expected_dim:
            raise ValueError("BPR checkpoint embedding dimension mismatch")
        if user_factors.shape[0] != len(metadata["user_ids"]):
            raise ValueError("BPR checkpoint user index mismatch")
        if item_factors.shape[0] != len(metadata["item_ids"]):
            raise ValueError("BPR checkpoint item index mismatch")

        return cls(
            user_index=IdIndex(tuple(metadata["user_ids"])),
            item_index=IdIndex(tuple(metadata["item_ids"])),
            user_factors=user_factors,
            item_factors=item_factors,
        )
