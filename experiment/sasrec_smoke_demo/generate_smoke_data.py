"""Generate the deterministic interaction data used by the SASRec smoke demo."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_SEED = 20260714
DEFAULT_USERS = 128
DEFAULT_ITEMS = 120
DEFAULT_INTERACTIONS_PER_USER = 24
CLUSTER_COUNT = 6


def build_rows(
    seed: int = DEFAULT_SEED,
    user_count: int = DEFAULT_USERS,
    item_count: int = DEFAULT_ITEMS,
    interactions_per_user: int = DEFAULT_INTERACTIONS_PER_USER,
) -> list[tuple[int, int, int]]:
    """Build ordered synthetic interactions with a learnable sequential pattern."""
    if user_count < 1:
        raise ValueError("user_count must be positive")
    if item_count < CLUSTER_COUNT:
        raise ValueError(f"item_count must be at least {CLUSTER_COUNT}")
    if interactions_per_user < 3:
        raise ValueError("interactions_per_user must be at least 3")

    rows: list[tuple[int, int, int]] = []
    items_per_cluster = item_count // CLUSTER_COUNT
    if items_per_cluster < 2:
        raise ValueError("each item cluster must contain at least two items")

    for user_id in range(1, user_count + 1):
        rng = random.Random(seed + user_id * 10_007)
        cluster = (user_id - 1) % CLUSTER_COUNT
        cluster_start = cluster * items_per_cluster + 1
        offset = (user_id * 3) % items_per_cluster

        for step in range(interactions_per_user):
            # Most interactions follow a cyclic preference pattern. Occasional
            # deterministic noise keeps the example from becoming trivial.
            if rng.random() < 0.85:
                local_item = (offset + step) % items_per_cluster
                item_id = cluster_start + local_item
            else:
                item_id = rng.randint(1, item_count)

            timestamp = 1_800_000_000 + user_id * 1_000 + step
            rows.append((user_id, item_id, timestamp))

    return rows


def write_dataset(
    output_path: Path,
    seed: int = DEFAULT_SEED,
    user_count: int = DEFAULT_USERS,
    item_count: int = DEFAULT_ITEMS,
    interactions_per_user: int = DEFAULT_INTERACTIONS_PER_USER,
) -> int:
    """Write the generated rows with stable LF line endings."""
    rows = build_rows(
        seed=seed,
        user_count=user_count,
        item_count=item_count,
        interactions_per_user=interactions_per_user,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("user_id", "item_id", "timestamp"))
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "smoke_interactions.csv",
        help="CSV output path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument(
        "--interactions-per-user",
        type=int,
        default=DEFAULT_INTERACTIONS_PER_USER,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_count = write_dataset(
        output_path=args.output,
        seed=args.seed,
        user_count=args.users,
        item_count=args.items,
        interactions_per_user=args.interactions_per_user,
    )
    print(f"Wrote {row_count} interactions to {args.output}")


if __name__ == "__main__":
    main()
