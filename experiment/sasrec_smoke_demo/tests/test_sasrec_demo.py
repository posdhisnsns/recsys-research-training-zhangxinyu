from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from generate_smoke_data import write_dataset  # noqa: E402

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from sasrec_demo import (  # noqa: E402
        SASRec,
        SASRecTrainingDataset,
        build_parser,
        load_interactions,
        run_training,
    )


class SmokeGeneratorTests(unittest.TestCase):
    def test_generator_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.assertEqual(write_dataset(first), 3072)
            self.assertEqual(write_dataset(second), 3072)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(len(first.read_text(encoding="utf-8").splitlines()), 3073)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SmokePipelineTests(unittest.TestCase):
    def test_split_boundaries_and_negative_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "smoke.csv"
            write_dataset(data_path)
            splits = load_interactions(data_path)

            self.assertEqual(splits.user_count, 128)
            self.assertEqual(splits.item_count, 120)
            self.assertEqual(splits.interaction_count, 3072)
            for user_id, full_sequence in splits.sequences.items():
                reconstructed = splits.train[user_id] + [
                    splits.valid[user_id],
                    splits.test[user_id],
                ]
                self.assertEqual(reconstructed, full_sequence)
                self.assertEqual(len(splits.train[user_id]), 22)

            dataset = SASRecTrainingDataset(splits, max_len=20, seed=20260714)
            user_id, _, positives, negatives = dataset[0]
            valid_positions = positives.ne(0)
            seen_items = set(splits.sequences[int(user_id)])
            for negative in negatives[valid_positions].tolist():
                self.assertNotIn(negative, seen_items)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SASRecModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260714)
        torch.set_num_threads(1)

    def make_model(self) -> SASRec:
        return SASRec(
            item_count=120,
            max_len=4,
            hidden_size=16,
            num_heads=2,
            num_blocks=1,
            dropout=0.0,
        )

    def test_output_shapes_and_causal_mask(self) -> None:
        model = self.make_model().eval()
        first_sequence = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        second_sequence = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)

        with torch.no_grad():
            first_features = model.encode(first_sequence)
            second_features = model.encode(second_sequence)
            all_scores = model.score_all_items(first_sequence)

        self.assertEqual(tuple(first_features.shape), (1, 4, 16))
        self.assertEqual(tuple(all_scores.shape), (1, 121))
        self.assertTrue(
            torch.allclose(
                first_features[:, 2, :],
                second_features[:, 2, :],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_one_cpu_update_has_finite_loss(self) -> None:
        model = self.make_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        sequences = torch.tensor(
            [[0, 1, 2, 3], [0, 5, 6, 7]], dtype=torch.long
        )
        positives = torch.tensor(
            [[0, 2, 3, 4], [0, 6, 7, 8]], dtype=torch.long
        )
        negatives = torch.tensor(
            [[0, 20, 21, 22], [0, 30, 31, 32]], dtype=torch.long
        )
        before = model.item_embedding.weight.detach().clone()

        positive_logits, negative_logits = model(
            sequences, positives, negatives
        )
        mask = positives.ne(0)
        loss = criterion(
            positive_logits[mask], torch.ones_like(positive_logits[mask])
        ) + criterion(
            negative_logits[mask], torch.zeros_like(negative_logits[mask])
        )
        self.assertTrue(math.isfinite(float(loss.detach())))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, model.item_embedding.weight.detach()))

    def test_cpu_end_to_end_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "smoke.csv"
            output_root = root / "outputs"
            write_dataset(data_path)
            args = build_parser().parse_args(
                [
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--max-len",
                    "20",
                    "--hidden-size",
                    "16",
                    "--num-heads",
                    "2",
                    "--num-blocks",
                    "1",
                    "--dropout",
                    "0",
                    "--data-path",
                    str(data_path),
                    "--output-root",
                    str(output_root),
                ]
            )
            run_dir = run_training(args)

            for filename in (
                "config.json",
                "train.log",
                "metrics.jsonl",
                "summary.json",
                "best_model.pt",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
