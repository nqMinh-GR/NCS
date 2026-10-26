"""Group splits. A group never occupies more than one of train, val, test."""

from __future__ import annotations

import random
from typing import Iterable, Sequence


def assign_splits(
    groups: dict[str, Sequence[str]],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    official_test_ids: Iterable[str] = (),
) -> dict[str, str]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    official = set(official_test_ids)
    official_groups = []
    free_groups = []
    for gid, members in groups.items():
        if official.intersection(members):
            official_groups.append(gid)
        else:
            free_groups.append(gid)
    free_groups = sorted(free_groups)
    rng = random.Random(seed)
    rng.shuffle(free_groups)
    n = len(free_groups)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    assignment = {gid: "test" for gid in official_groups}
    for index, gid in enumerate(free_groups):
        if index < n_train:
            assignment[gid] = "train"
        elif index < n_train + n_val:
            assignment[gid] = "val"
        else:
            assignment[gid] = "test"
    return assignment


def sample_ids_for_split(
    groups: dict[str, Sequence[str]],
    assignment: dict[str, str],
    split: str,
) -> list[str]:
    ids: list[str] = []
    for gid, members in groups.items():
        if assignment.get(gid) == split:
            ids.extend(members)
    return ids


def leakage(groups: dict[str, Sequence[str]], assignment: dict[str, str]) -> list[str]:
    problems = []
    seen: dict[str, str] = {}
    for gid, members in groups.items():
        split = assignment[gid]
        for sample_id in members:
            previous = seen.get(sample_id)
            if previous and previous != split:
                problems.append(sample_id)
            seen[sample_id] = split
    return problems


def excluded_official_duplicates(
    groups: dict[str, Sequence[str]],
    assignment: dict[str, str],
    official_test_ids: Iterable[str],
) -> list[dict[str, str]]:
    official = set(official_test_ids)
    rows = []
    for gid, members in groups.items():
        if assignment.get(gid) != "test":
            continue
        if not official.intersection(members):
            continue
        for sample_id in members:
            if sample_id not in official:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "group_id": gid,
                        "reason": "duplicate_of_official_test",
                    }
                )
    return rows


def calibration_ids(train_ids: Sequence[str], n: int, seed: int) -> list[str]:
    pool = list(train_ids)
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n >= len(pool):
        return pool
    return pool[:n]
