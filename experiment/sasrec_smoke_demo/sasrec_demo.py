"""A compact, classroom-oriented SASRec training and evaluation program.

The implementation follows the SASRec architecture and training pattern while
keeping the complete smoke-demo pipeline in one file. It is informed by:

- https://github.com/kang205/SASRec
- https://github.com/pmixer/SASRec.pytorch

The code is an educational reimplementation rather than a benchmark replica.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


MIB = 1024 * 1024


@dataclass(frozen=True)
class SequenceSplits:
    """Contiguous integer IDs and leave-two-out sequence splits."""

    sequences: dict[int, list[int]]
    train: dict[int, list[int]]
    valid: dict[int, int]
    test: dict[int, int]
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]

    @property
    def user_count(self) -> int:
        return len(self.sequences)

    @property
    def item_count(self) -> int:
        return len(self.item_mapping)

    @property
    def interaction_count(self) -> int:
        return sum(len(sequence) for sequence in self.sequences.values())


def identifier_sort_key(value: str) -> tuple[int, int | str]:
    """Sort numeric identifiers numerically and other identifiers lexically."""
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_interactions(data_path: Path) -> SequenceSplits:
    """Read CSV interactions, remap IDs, and hold out two events per user."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_events: dict[str, list[tuple[int, int, str]]] = {}
    raw_items: set[str] = set()

    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"user_id", "item_id", "timestamp"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_fields:
            raise ValueError(
                "CSV fields must be exactly: user_id,item_id,timestamp"
            )

        for row_number, row in enumerate(reader, start=2):
            user_id = row["user_id"].strip()
            item_id = row["item_id"].strip()
            if not user_id or not item_id:
                raise ValueError(f"Empty user_id or item_id on row {row_number}")
            try:
                timestamp = int(row["timestamp"])
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp on row {row_number}") from exc

            raw_events.setdefault(user_id, []).append(
                (timestamp, row_number, item_id)
            )
            raw_items.add(item_id)

    if not raw_events:
        raise ValueError("The interaction file is empty")

    user_mapping = {
        raw_id: index
        for index, raw_id in enumerate(
            sorted(raw_events, key=identifier_sort_key), start=1
        )
    }
    item_mapping = {
        raw_id: index
        for index, raw_id in enumerate(
            sorted(raw_items, key=identifier_sort_key), start=1
        )
    }

    sequences: dict[int, list[int]] = {}
    train: dict[int, list[int]] = {}
    valid: dict[int, int] = {}
    test: dict[int, int] = {}

    for raw_user, events in raw_events.items():
        user_id = user_mapping[raw_user]
        ordered_items = [
            item_mapping[item_id]
            for _, _, item_id in sorted(events, key=lambda event: (event[0], event[1]))
        ]
        if len(ordered_items) < 3:
            raise ValueError(
                f"User {raw_user!r} has fewer than three interactions"
            )
        sequences[user_id] = ordered_items
        train[user_id] = ordered_items[:-2]
        valid[user_id] = ordered_items[-2]
        test[user_id] = ordered_items[-1]

    return SequenceSplits(
        sequences=sequences,
        train=train,
        valid=valid,
        test=test,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def right_align(values: Sequence[int], max_len: int) -> list[int]:
    """Truncate from the left and zero-pad on the left."""
    trimmed = list(values[-max_len:])
    return [0] * (max_len - len(trimmed)) + trimmed


class SASRecTrainingDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Create one canonical next-item training sample per user."""

    def __init__(self, splits: SequenceSplits, max_len: int, seed: int) -> None:
        self.splits = splits
        self.max_len = max_len
        self.seed = seed
        self.users = sorted(splits.train)
        catalog = set(range(1, splits.item_count + 1))
        self.negative_candidates: dict[int, tuple[int, ...]] = {}

        for user_id in self.users:
            candidates = tuple(sorted(catalog - set(splits.sequences[user_id])))
            if not candidates:
                raise ValueError(f"User {user_id} has no available negative items")
            self.negative_candidates[user_id] = candidates

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        user_id = self.users[index]
        training_sequence = self.splits.train[user_id]
        input_items = training_sequence[:-1]
        positive_items = training_sequence[1:]

        rng = random.Random(self.seed + user_id * 1_000_003)
        candidates = self.negative_candidates[user_id]
        negative_items = [rng.choice(candidates) for _ in positive_items]

        sequences = torch.tensor(
            right_align(input_items, self.max_len), dtype=torch.long
        )
        positives = torch.tensor(
            right_align(positive_items, self.max_len), dtype=torch.long
        )
        negatives = torch.tensor(
            right_align(negative_items, self.max_len), dtype=torch.long
        )
        return torch.tensor(user_id), sequences, positives, negatives


class PointWiseFeedForward(nn.Module):
    """Position-wise feed-forward layer used inside each SASRec block."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.linear1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        return self.dropout2(outputs)


class SASRecBlock(nn.Module):
    """Pre-normalized causal self-attention and feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.feed_forward = PointWiseFeedForward(hidden_size, dropout)

    def forward(
        self,
        inputs: Tensor,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.attention_norm(inputs)
        attention_outputs, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        outputs = inputs + self.attention_dropout(attention_outputs)
        outputs = outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        outputs = outputs + self.feed_forward(
            self.feed_forward_norm(outputs)
        )
        return outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""

    def __init__(
        self,
        item_count: int,
        max_len: int,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.item_count = item_count
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.item_embedding = nn.Embedding(
            item_count + 1, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            max_len + 1, hidden_size, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(hidden_size, num_heads, dropout)
            for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, sequences: Tensor) -> Tensor:
        """Encode right-aligned item sequences without attending to the future."""
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape [batch, sequence_length]")
        if sequences.shape[1] > self.max_len:
            raise ValueError("sequence length exceeds the configured max_len")

        padding_mask = sequences.eq(0)
        sequence_length = sequences.shape[1]
        positions = torch.arange(
            1, sequence_length + 1, device=sequences.device
        ).unsqueeze(0)
        positions = positions.expand_as(sequences)
        positions = positions.masked_fill(padding_mask, 0)

        outputs = self.item_embedding(sequences) * math.sqrt(self.hidden_size)
        outputs = outputs + self.position_embedding(positions)
        outputs = self.embedding_dropout(outputs)
        outputs = outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=sequences.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            outputs = block(outputs, causal_mask, padding_mask)

        outputs = self.final_norm(outputs)
        return outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        sequences: Tensor,
        positive_items: Tensor,
        negative_items: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = self.encode(sequences)
        positive_embeddings = self.item_embedding(positive_items)
        negative_embeddings = self.item_embedding(negative_items)
        positive_logits = (features * positive_embeddings).sum(dim=-1)
        negative_logits = (features * negative_embeddings).sum(dim=-1)
        return positive_logits, negative_logits

    def score_all_items(self, sequences: Tensor) -> Tensor:
        """Score the complete catalog from the final sequence position."""
        final_features = self.encode(sequences)[:, -1, :]
        return final_features @ self.item_embedding.weight.transpose(0, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    return torch.device("cpu")


def configure_cuda_allocator(device: torch.device, limit_mib: int) -> float | None:
    """Limit PyTorch's caching allocator before model tensors are created."""
    if device.type != "cuda":
        return None
    device_index = device.index or 0
    torch.cuda.set_device(device_index)
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    fraction = limit_mib * MIB / total_bytes
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"allocator limit must be between 1 MiB and {total_bytes // MIB} MiB"
        )
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    return fraction


def query_process_gpu_memory_mib(pid: int | None = None) -> float | None:
    """Return this process's nvidia-smi memory total across visible GPUs."""
    target_pid = pid or os.getpid()
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    memory_values: list[float] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            process_pid = int(parts[0])
            memory_mib = float(parts[1].split()[0])
        except (ValueError, IndexError):
            continue
        if process_pid == target_pid:
            memory_values.append(memory_mib)
    return sum(memory_values) if memory_values else None


class GPUMemoryMonitor:
    """Track PyTorch peaks and enforce the nvidia-smi process budget."""

    def __init__(
        self,
        device: torch.device,
        budget_mib: int,
        logger: logging.Logger,
    ) -> None:
        self.device = device
        self.budget_mib = budget_mib
        self.logger = logger
        self.peak_process_mib: float | None = None
        self.peak_allocated_mib = 0.0
        self.peak_reserved_mib = 0.0

    def sample(self, label: str) -> dict[str, float | None]:
        if self.device.type != "cuda":
            return {
                "process_mib": None,
                "torch_allocated_mib": None,
                "torch_reserved_mib": None,
            }

        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.max_memory_allocated(self.device) / MIB
        reserved = torch.cuda.max_memory_reserved(self.device) / MIB
        process_memory = query_process_gpu_memory_mib()
        self.peak_allocated_mib = max(self.peak_allocated_mib, allocated)
        self.peak_reserved_mib = max(self.peak_reserved_mib, reserved)
        if process_memory is not None:
            self.peak_process_mib = max(
                self.peak_process_mib or 0.0, process_memory
            )

        process_text = (
            f"{process_memory:.0f} MiB"
            if process_memory is not None
            else "unavailable"
        )
        self.logger.info(
            "GPU memory [%s]: process=%s, torch_allocated=%.1f MiB, "
            "torch_reserved=%.1f MiB",
            label,
            process_text,
            allocated,
            reserved,
        )
        if process_memory is not None and process_memory > self.budget_mib:
            raise RuntimeError(
                f"GPU memory budget exceeded: {process_memory:.0f} MiB > "
                f"{self.budget_mib} MiB"
            )
        return {
            "process_mib": process_memory,
            "torch_allocated_mib": allocated,
            "torch_reserved_mib": reserved,
        }

    def summary(self) -> dict[str, float | None]:
        return {
            "peak_process_mib": self.peak_process_mib,
            "peak_torch_allocated_mib": (
                self.peak_allocated_mib if self.device.type == "cuda" else None
            ),
            "peak_torch_reserved_mib": (
                self.peak_reserved_mib if self.device.type == "cuda" else None
            ),
        }


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
    }
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        snapshot.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": round(properties.total_memory / MIB),
                "visible_cuda_devices": torch.cuda.device_count(),
            }
        )
    return snapshot


def batch_contexts(contexts: Sequence[Sequence[int]], max_len: int) -> Tensor:
    return torch.tensor(
        [right_align(context, max_len) for context in contexts],
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate(
    model: SASRec,
    splits: SequenceSplits,
    device: torch.device,
    stage: str,
    max_len: int,
    batch_size: int,
    top_k: int,
) -> dict[str, float]:
    """Evaluate validation or test targets against the complete item catalog."""
    if stage not in {"valid", "test"}:
        raise ValueError("stage must be 'valid' or 'test'")

    users = sorted(splits.train)
    contexts: list[list[int]] = []
    targets: list[int] = []
    for user_id in users:
        if stage == "valid":
            contexts.append(list(splits.train[user_id]))
            targets.append(splits.valid[user_id])
        else:
            contexts.append(splits.train[user_id] + [splits.valid[user_id]])
            targets.append(splits.test[user_id])

    model.eval()
    hits = 0.0
    ndcg = 0.0
    effective_k = min(top_k, splits.item_count)

    for start in range(0, len(users), batch_size):
        batch_context = contexts[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        sequences = batch_contexts(batch_context, max_len).to(device)
        scores = model.score_all_items(sequences)
        scores[:, 0] = -torch.inf

        for row_index, (context, target) in enumerate(
            zip(batch_context, batch_targets)
        ):
            target_score = scores[row_index, target].clone()
            seen_items = sorted(set(context))
            if seen_items:
                scores[row_index, seen_items] = -torch.inf
            scores[row_index, target] = target_score

        recommendations = torch.topk(
            scores, k=effective_k, dim=1
        ).indices.cpu()
        for row_index, target in enumerate(batch_targets):
            matches = (recommendations[row_index] == target).nonzero(
                as_tuple=False
            )
            if matches.numel() > 0:
                rank = int(matches[0, 0])
                hits += 1.0
                ndcg += 1.0 / math.log2(rank + 2)

    user_count = len(users)
    return {"hr": hits / user_count, "ndcg": ndcg / user_count}


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"sasrec_demo.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        run_dir / "train.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "max_len",
        "hidden_size",
        "num_heads",
        "num_blocks",
        "top_k",
        "allocator_limit_mib",
        "gpu_budget_mib",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.allocator_limit_mib >= args.gpu_budget_mib:
        raise ValueError(
            "--allocator-limit-mib must be smaller than --gpu-budget-mib"
        )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run_training(args: argparse.Namespace) -> Path:
    validate_args(args)
    device = resolve_device(args.device)
    allocator_fraction = configure_cuda_allocator(
        device, args.allocator_limit_mib
    )
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = create_run_directory(args.output_root)
    logger = setup_logger(run_dir)
    try:
        config = json_ready_args(args)
        config["resolved_device"] = str(device)
        config["allocator_fraction"] = allocator_fraction
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        environment = environment_snapshot(device)
        logger.info("Environment: %s", json.dumps(environment, ensure_ascii=False))
        splits = load_interactions(args.data_path)
        logger.info(
            "Data: users=%d, items=%d, interactions=%d",
            splits.user_count,
            splits.item_count,
            splits.interaction_count,
        )

        dataset = SASRecTrainingDataset(splits, args.max_len, args.seed)
        loader_generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=loader_generator,
        )
        model = SASRec(
            item_count=splits.item_count,
            max_len=args.max_len,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        ).to(device)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        logger.info("Model parameters: %d", parameter_count)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98)
        )
        criterion = nn.BCEWithLogitsLoss()
        monitor = GPUMemoryMonitor(device, args.gpu_budget_mib, logger)
        monitor.sample("model initialized")

        best_score = (-1.0, -1.0)
        best_epoch = 0
        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "metrics.jsonl"

        with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_file:
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_total = 0.0
                batch_count = 0

                for _, sequences, positives, negatives in loader:
                    sequences = sequences.to(device)
                    positives = positives.to(device)
                    negatives = negatives.to(device)
                    positive_logits, negative_logits = model(
                        sequences, positives, negatives
                    )
                    valid_positions = positives.ne(0)
                    positive_loss = criterion(
                        positive_logits[valid_positions],
                        torch.ones_like(positive_logits[valid_positions]),
                    )
                    negative_loss = criterion(
                        negative_logits[valid_positions],
                        torch.zeros_like(negative_logits[valid_positions]),
                    )
                    loss = positive_loss + negative_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Encountered a non-finite training loss")

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.detach().cpu())
                    batch_count += 1

                average_loss = loss_total / batch_count
                valid_metrics = evaluate(
                    model=model,
                    splits=splits,
                    device=device,
                    stage="valid",
                    max_len=args.max_len,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                )
                memory = monitor.sample(f"epoch {epoch}")
                record = {
                    "epoch": epoch,
                    "train_loss": average_loss,
                    "valid_hr": valid_metrics["hr"],
                    "valid_ndcg": valid_metrics["ndcg"],
                    **memory,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                logger.info(
                    "Epoch %02d | loss=%.4f | valid HR@%d=%.4f | "
                    "valid NDCG@%d=%.4f",
                    epoch,
                    average_loss,
                    args.top_k,
                    valid_metrics["hr"],
                    args.top_k,
                    valid_metrics["ndcg"],
                )

                epoch_score = (valid_metrics["ndcg"], valid_metrics["hr"])
                if epoch_score > best_score:
                    best_score = epoch_score
                    best_epoch = epoch
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "item_count": splits.item_count,
                            "config": config,
                            "best_epoch": best_epoch,
                        },
                        checkpoint_path,
                    )

        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_valid = evaluate(
            model,
            splits,
            device,
            "valid",
            args.max_len,
            args.batch_size,
            args.top_k,
        )
        final_test = evaluate(
            model,
            splits,
            device,
            "test",
            args.max_len,
            args.batch_size,
            args.top_k,
        )
        monitor.sample("final evaluation")
        memory_summary = monitor.summary()
        process_peak = memory_summary["peak_process_mib"]
        summary = {
            "run_directory": str(run_dir),
            "data": {
                "users": splits.user_count,
                "items": splits.item_count,
                "interactions": splits.interaction_count,
            },
            "model_parameters": parameter_count,
            "best_epoch": best_epoch,
            "top_k": args.top_k,
            "validation": final_valid,
            "test": final_test,
            "gpu_memory_budget_mib": args.gpu_budget_mib,
            "gpu_memory": memory_summary,
            "gpu_memory_within_budget": (
                None
                if process_peak is None
                else process_peak <= args.gpu_budget_mib
            ),
            "environment": environment,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Best epoch %d | test HR@%d=%.4f | test NDCG@%d=%.4f",
            best_epoch,
            args.top_k,
            final_test["hr"],
            args.top_k,
            final_test["ndcg"],
        )
        logger.info("Outputs: %s", run_dir)
        return run_dir
    finally:
        close_logger(logger)


def run_environment_check(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    allocator_fraction = configure_cuda_allocator(
        device, args.allocator_limit_mib
    )
    snapshot = environment_snapshot(device)
    snapshot["allocator_limit_mib"] = args.allocator_limit_mib
    snapshot["allocator_fraction"] = allocator_fraction
    if device.type == "cuda":
        process_memory = query_process_gpu_memory_mib()
        snapshot["nvidia_smi_process_memory_mib"] = process_memory
        if process_memory is not None and process_memory > args.gpu_budget_mib:
            raise RuntimeError(
                f"GPU memory budget exceeded during environment check: "
                f"{process_memory:.0f} MiB > {args.gpu_budget_mib} MiB"
            )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=script_dir / "data" / "smoke_interactions.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=script_dir / "outputs"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allocator-limit-mib", type=int, default=700)
    parser.add_argument("--gpu-budget-mib", type=int, default=1024)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_only:
        run_environment_check(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
