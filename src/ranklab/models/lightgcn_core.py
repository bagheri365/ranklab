"""Deterministic NumPy LightGCN core.

This module intentionally lives beside any scaffold placeholder in
``ranklab.models.lightgcn``. M0.9a freezes and tests the numerical graph model
before wiring it into the public training surface.

LightGCN reference semantics implemented here:

* bipartite graph edges come from positive user-item pairs only;
* symmetric degree normalization is used;
* propagation has no feature transform and no nonlinear activation;
* the scoring representation is the mean of layer 0 through layer L;
* checkpoints contain only base embeddings and graph/index metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import numpy as np

from ranklab.models.bpr_mf import IdIndex


Id = Hashable
Pair = tuple[Id, Id]


@dataclass(frozen=True)
class LightGCNGraph:
    """Indexed positive bipartite graph with normalized edge weights."""

    user_indices: np.ndarray
    item_indices: np.ndarray
    edge_weights: np.ndarray
    user_degrees: np.ndarray
    item_degrees: np.ndarray

    @classmethod
    def from_positive_pairs(
        cls,
        positive_pairs: Sequence[Pair],
        *,
        user_index: IdIndex,
        item_index: IdIndex,
    ) -> "LightGCNGraph":
        if not positive_pairs:
            raise ValueError("at least one positive graph edge is required")

        user_map = user_index.to_index
        item_map = item_index.to_index

        # Graph semantics are pair-collapsed: duplicate positive pairs must not
        # create multi-edge weighting.
        unique_pairs = sorted(set(positive_pairs), key=lambda p: (repr(p[0]), repr(p[1])))

        try:
            users = np.fromiter(
                (user_map[u] for u, _ in unique_pairs),
                dtype=np.int64,
                count=len(unique_pairs),
            )
            items = np.fromiter(
                (item_map[i] for _, i in unique_pairs),
                dtype=np.int64,
                count=len(unique_pairs),
            )
        except KeyError as exc:
            raise KeyError(f"graph pair references unknown id: {exc.args[0]!r}") from exc

        user_degrees = np.bincount(users, minlength=len(user_index.values)).astype(np.float64)
        item_degrees = np.bincount(items, minlength=len(item_index.values)).astype(np.float64)

        if np.any(user_degrees[users] <= 0) or np.any(item_degrees[items] <= 0):
            raise ValueError("invalid zero degree on observed graph edge")

        weights = 1.0 / np.sqrt(user_degrees[users] * item_degrees[items])

        return cls(
            user_indices=users,
            item_indices=items,
            edge_weights=weights.astype(np.float64),
            user_degrees=user_degrees,
            item_degrees=item_degrees,
        )


@dataclass
class LightGCNModel:
    """Base embeddings plus a frozen positive interaction graph."""

    user_index: IdIndex
    item_index: IdIndex
    graph: LightGCNGraph
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    num_layers: int

    @classmethod
    def initialize(
        cls,
        *,
        user_ids: Iterable[Id],
        item_ids: Iterable[Id],
        positive_pairs: Sequence[Pair],
        embedding_dim: int,
        num_layers: int,
        seed: int,
        init_std: float = 0.01,
    ) -> "LightGCNModel":
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        user_index = IdIndex.from_values(user_ids)
        item_index = IdIndex.from_values(item_ids)
        if not user_index.values or not item_index.values:
            raise ValueError("at least one user and one item are required")

        graph = LightGCNGraph.from_positive_pairs(
            positive_pairs,
            user_index=user_index,
            item_index=item_index,
        )

        rng = np.random.default_rng(seed)
        user_embeddings = rng.normal(
            0.0, init_std, size=(len(user_index.values), embedding_dim)
        ).astype(np.float64)
        item_embeddings = rng.normal(
            0.0, init_std, size=(len(item_index.values), embedding_dim)
        ).astype(np.float64)

        return cls(
            user_index=user_index,
            item_index=item_index,
            graph=graph,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            num_layers=num_layers,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.user_embeddings.shape[1])

    def _propagate_once(
        self,
        users: np.ndarray,
        items: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        next_users = np.zeros_like(users)
        next_items = np.zeros_like(items)
        weighted_items = self.graph.edge_weights[:, None] * items[self.graph.item_indices]
        weighted_users = self.graph.edge_weights[:, None] * users[self.graph.user_indices]
        np.add.at(next_users, self.graph.user_indices, weighted_items)
        np.add.at(next_items, self.graph.item_indices, weighted_users)
        return next_users, next_items

    def propagated_layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
        layers = [(self.user_embeddings.copy(), self.item_embeddings.copy())]
        users = self.user_embeddings
        items = self.item_embeddings
        for _ in range(self.num_layers):
            users, items = self._propagate_once(users, items)
            layers.append((users, items))
        return layers

    def final_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        layers = self.propagated_layers()
        users = np.mean(np.stack([layer[0] for layer in layers], axis=0), axis=0)
        items = np.mean(np.stack([layer[1] for layer in layers], axis=0), axis=0)
        return users, items

    def score(self, user_id: Id, item_id: Id) -> float:
        users, items = self.final_embeddings()
        u = self.user_index.encode(user_id)
        i = self.item_index.encode(item_id)
        return float(np.dot(users[u], items[i]))

    def score_items(self, user_id: Id, item_ids: Sequence[Id]) -> np.ndarray:
        users, items = self.final_embeddings()
        u = self.user_index.encode(user_id)
        item_map = self.item_index.to_index
        idx = np.fromiter(
            (item_map[item_id] for item_id in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        return items[idx] @ users[u]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "ranklab_lightgcn_npz_v1",
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "user_ids": list(self.user_index.values),
            "item_ids": list(self.item_index.values),
        }
        np.savez_compressed(
            destination,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            graph_user_indices=self.graph.user_indices,
            graph_item_indices=self.graph.item_indices,
            graph_edge_weights=self.graph.edge_weights,
            graph_user_degrees=self.graph.user_degrees,
            graph_item_degrees=self.graph.item_degrees,
            metadata=np.array(json.dumps(metadata, separators=(",", ":"))),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LightGCNModel":
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            if metadata.get("format") != "ranklab_lightgcn_npz_v1":
                raise ValueError("unsupported LightGCN checkpoint format")

            user_embeddings = np.asarray(checkpoint["user_embeddings"], dtype=np.float64)
            item_embeddings = np.asarray(checkpoint["item_embeddings"], dtype=np.float64)
            graph = LightGCNGraph(
                user_indices=np.asarray(checkpoint["graph_user_indices"], dtype=np.int64),
                item_indices=np.asarray(checkpoint["graph_item_indices"], dtype=np.int64),
                edge_weights=np.asarray(checkpoint["graph_edge_weights"], dtype=np.float64),
                user_degrees=np.asarray(checkpoint["graph_user_degrees"], dtype=np.float64),
                item_degrees=np.asarray(checkpoint["graph_item_degrees"], dtype=np.float64),
            )

        user_index = IdIndex(tuple(metadata["user_ids"]))
        item_index = IdIndex(tuple(metadata["item_ids"]))
        dim = int(metadata["embedding_dim"])

        if user_embeddings.shape != (len(user_index.values), dim):
            raise ValueError("LightGCN checkpoint user embedding shape mismatch")
        if item_embeddings.shape != (len(item_index.values), dim):
            raise ValueError("LightGCN checkpoint item embedding shape mismatch")

        return cls(
            user_index=user_index,
            item_index=item_index,
            graph=graph,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            num_layers=int(metadata["num_layers"]),
        )
