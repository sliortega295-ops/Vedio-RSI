from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from rolloutbench.events import (
    CorruptLedgerError,
    EventLedger,
    IdempotencyConflictError,
    atomic_write_stage_output,
)


def _append_many(path: str, worker: int, count: int) -> None:
    ledger = EventLedger(Path(path))
    for index in range(count):
        ledger.append(
            "episode_released",
            {"episode_id": f"W{worker}-{index}"},
            idempotency_key=f"release:{worker}:{index}",
        )


class EventLedgerTests(unittest.TestCase):
    def test_concurrent_append_has_monotonic_unique_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            processes = [
                multiprocessing.Process(target=_append_many, args=(str(path), worker, 25))
                for worker in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(0, process.exitcode)

            events = EventLedger(path).read()
            self.assertEqual(100, len(events))
            self.assertEqual(list(range(1, 101)), [event["sequence"] for event in events])
            self.assertEqual(100, len({event["event_id"] for event in events}))
            self.assertTrue(all(event["utc"] and event["monotonic_ns"] > 0 for event in events))
            self.assertEqual(
                sorted(event["monotonic_ns"] for event in events),
                [event["monotonic_ns"] for event in events],
            )
            self.assertEqual(1, len({event["boot_id"] for event in events}))

    def test_tail_half_line_is_truncated_but_middle_corruption_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            ledger = EventLedger(path)
            ledger.append("run_started", {"run_id": "r"}, idempotency_key="run:r")
            with path.open("ab") as handle:
                handle.write(b'{"sequence": 2, "event_type":')

            events = ledger.read()
            self.assertEqual(1, len(events))
            self.assertTrue(path.read_bytes().endswith(b"\n"))

            valid = path.read_bytes()
            path.write_bytes(valid + b"not-json\n" + valid)
            with self.assertRaises(CorruptLedgerError):
                ledger.read()

    def test_idempotent_replay_and_conflict_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EventLedger(Path(directory) / "events.jsonl")
            first = ledger.append(
                "episode_released", {"episode_id": "K01"}, idempotency_key="release:K01"
            )
            replay = ledger.append(
                "episode_released", {"episode_id": "K01"}, idempotency_key="release:K01"
            )
            self.assertEqual(first, replay)
            self.assertEqual(1, len(ledger.read()))
            with self.assertRaises(IdempotencyConflictError):
                ledger.append(
                    "episode_released", {"episode_id": "K02"}, idempotency_key="release:K01"
                )

            decision = {"outcome": "retain", "frontier_status": "new_best"}
            ledger.seal_decision("K01", decision)
            ledger.seal_decision("K01", decision)
            self.assertEqual(1, len(ledger.reconstruct().decisions))
            with self.assertRaises(IdempotencyConflictError):
                ledger.seal_decision("K01", {"outcome": "reject"})

    def test_stage_output_is_durable_before_completion_event_and_resume_marks_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = EventLedger(root / "events.jsonl")
            ledger.append(
                "stage_started",
                {"episode_id": "K20", "stage": "generate", "attempt": 1},
                idempotency_key="stage:K20:generate:1:start",
            )

            def crash_after_replace(point: str) -> None:
                if point == "after_replace":
                    raise RuntimeError("injected crash")

            output = root / "outputs" / "K20.json"
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                atomic_write_stage_output(
                    ledger,
                    output,
                    b'{"ok": true}\n',
                    episode_id="K20",
                    stage="generate",
                    attempt=1,
                    fault_hook=crash_after_replace,
                )
            self.assertEqual(b'{"ok": true}\n', output.read_bytes())
            state = ledger.reconstruct()
            self.assertEqual({("K20", "generate", 1)}, state.interrupted_stages)
            self.assertFalse(any(event["event_type"] == "stage_completed" for event in state.events))

            completed = atomic_write_stage_output(
                ledger,
                output,
                b'{"ok": true}\n',
                episode_id="K20",
                stage="generate",
                attempt=2,
            )
            self.assertEqual("stage_completed", completed["event_type"])
            self.assertEqual(64, len(completed["payload"]["output_sha256"]))

            replay = atomic_write_stage_output(
                ledger,
                output,
                b'{"ok": true}\n',
                episode_id="K20",
                stage="generate",
                attempt=2,
            )
            self.assertEqual(completed, replay)
            with self.assertRaises(IdempotencyConflictError):
                atomic_write_stage_output(
                    ledger,
                    output,
                    b'{"ok": false}\n',
                    episode_id="K20",
                    stage="generate",
                    attempt=2,
                )
            self.assertEqual(b'{"ok": true}\n', output.read_bytes())

    def test_crash_before_rename_leaves_no_committed_output_and_decision_is_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = EventLedger(root / "events.jsonl")
            output = root / "result.bin"

            def crash_before_replace(point: str) -> None:
                if point == "before_replace":
                    raise RuntimeError("before rename")

            with self.assertRaisesRegex(RuntimeError, "before rename"):
                atomic_write_stage_output(
                    ledger,
                    output,
                    b"candidate",
                    episode_id="C12",
                    stage="quality_v1",
                    attempt=1,
                    fault_hook=crash_before_replace,
                )
            self.assertFalse(output.exists())

            payload = {"outcome": "retain", "frontier_status": "quality_v1"}
            sealed = ledger.seal_decision("C12", payload)
            replay = ledger.seal_decision("C12", payload)
            self.assertEqual(sealed, replay)
            self.assertEqual(
                1,
                sum(event["event_type"] == "decision_sealed" for event in ledger.read()),
            )


if __name__ == "__main__":
    unittest.main()
