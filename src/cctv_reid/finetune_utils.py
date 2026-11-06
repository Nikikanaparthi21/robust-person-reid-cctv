"""Torch-free helpers for OSNet fine-tuning: identity split and P x K batch sampling (tested)."""

from __future__ import annotations

import random
from collections import defaultdict


def split_identities(pids: list[int], val_fraction: float = 0.1, seed: int = 255) -> tuple[set[int], set[int]]:
    """Deterministic train/validation identity split. Validation identities are never trained on."""
    ids = sorted({p for p in pids if p > 0})
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return set(ids[n_val:]), set(ids[:n_val])


def pk_batches(labels: list[int], p: int, k: int, n_batches: int, seed: int = 255):
    """Yield `n_batches` lists of dataset indices: P random identities x K images each.

    Identities with fewer than K images are sampled with replacement, so every batch has the
    P*K shape that batch-hard triplet loss needs.
    """
    by_id: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_id[y].append(i)
    ids = sorted(by_id)
    if len(ids) < p:
        raise ValueError(f"need at least {p} identities, have {len(ids)}")
    rng = random.Random(seed)
    for _ in range(n_batches):
        batch: list[int] = []
        for y in rng.sample(ids, p):
            pool = by_id[y]
            batch += rng.sample(pool, k) if len(pool) >= k else rng.choices(pool, k=k)
        yield batch
