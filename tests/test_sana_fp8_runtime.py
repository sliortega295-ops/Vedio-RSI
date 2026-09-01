from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_fp8.py"
)


def _load_module():
    module_name = "_sol_agent_test_sana_fp8"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fp8 = _load_module()


class FakeFP8Backend:
    """Shape-faithful CPU backend used to test wiring without claiming FP8."""

    name = "fake_shape_backend"

    def preflight(self, device: torch.device, dtype: torch.dtype):
        return {"backend": self.name, "device": str(device), "source_dtype": str(dtype)}

    def pack_weight(self, weight: torch.Tensor):
        return weight.t().contiguous(), torch.ones(
            1, weight.shape[0], dtype=torch.float32, device=weight.device
        )

    def linear(self, x, weight, weight_scale, bias):
        del weight_scale
        output = torch.matmul(x, weight)
        return output if bias is None else output + bias


class DummyFF(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 8):
        super().__init__()
        self.conv_inverted = nn.Conv2d(channels, hidden, 1)
        self.conv_point = nn.Conv2d(hidden, channels, 1)
        self.conv_depth = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self._fused_gate = False


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = DummyFF()


class DummyTransformer(nn.Module):
    def __init__(self, blocks: int = 3):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(DummyBlock() for _ in range(blocks))


class SanaFP8RuntimeTest(unittest.TestCase):
    def test_pointwise_adapter_preserves_conv_shape_and_values(self):
        torch.manual_seed(7)
        conv = nn.Conv2d(3, 5, 1, bias=True)
        x = torch.randn(2, 3, 4, 6).contiguous(memory_format=torch.channels_last)

        candidate = fp8.SanaFP8PointwiseConv2d.from_conv(
            conv, module_name="test.conv", backend=FakeFP8Backend()
        )
        expected = conv(x)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            actual = candidate(x)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(candidate.fp8_calls, 1)
        self.assertIn("SANA_FP8_MODULE_ACTIVE", stdout.getvalue())

    def test_off_policy_is_identity_and_does_not_construct_backend(self):
        model = DummyTransformer()
        before = [id(block.ff.conv_inverted) for block in model.transformer_blocks]

        report = fp8.install_sana_fp8(
            model, policy=fp8.SanaFP8Policy(enabled=False)
        )

        self.assertEqual(report.status, "off")
        self.assertEqual(report.converted_modules, [])
        self.assertEqual(
            [id(block.ff.conv_inverted) for block in model.transformer_blocks],
            before,
        )

    def test_block_guard_converts_only_selected_pointwise_projections(self):
        model = DummyTransformer()
        report = fp8.install_sana_fp8(
            model,
            policy=fp8.SanaFP8Policy(enabled=True, block_start=1, block_end=1),
            backend=FakeFP8Backend(),
        )

        self.assertEqual(report.status, "installed")
        self.assertEqual(
            report.converted_modules,
            [
                "transformer_blocks.1.ff.conv_inverted",
                "transformer_blocks.1.ff.conv_point",
            ],
        )
        self.assertIsInstance(model.transformer_blocks[0].ff.conv_inverted, nn.Conv2d)
        self.assertIsInstance(
            model.transformer_blocks[1].ff.conv_inverted,
            fp8.SanaFP8PointwiseConv2d,
        )
        self.assertIsInstance(
            model.transformer_blocks[1].ff.conv_point,
            fp8.SanaFP8PointwiseConv2d,
        )
        self.assertIsInstance(model.transformer_blocks[1].ff.conv_depth, nn.Conv2d)
        self.assertIsInstance(model.transformer_blocks[2].ff.conv_point, nn.Conv2d)
        self.assertEqual(len(report.skipped_modules), 4)

    def test_invalid_policy_fails_closed(self):
        policies = [
            fp8.SanaFP8Policy(enabled=True, scope="all_linear"),
            fp8.SanaFP8Policy(enabled=True, block_start=-1),
            fp8.SanaFP8Policy(enabled=True, block_start=2, block_end=1),
            fp8.SanaFP8Policy(enabled=True, block_start=0, block_end=3),
        ]
        for policy in policies:
            with self.subTest(policy=policy), self.assertRaises(RuntimeError):
                fp8.install_sana_fp8(
                    DummyTransformer(), policy=policy, backend=FakeFP8Backend()
                )


if __name__ == "__main__":
    unittest.main()
