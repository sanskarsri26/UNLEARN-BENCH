import os
import unittest
from unittest.mock import patch

import torch

from unlearn_bench.utils.device import (
    DeviceUnavailableError,
    PrecisionUnavailableError,
    _resolve_device,
    select_device,
)


class DeviceTests(unittest.TestCase):
    def test_explicit_cpu_and_dtype(self):
        context = select_device("cpu", "fp32")
        self.assertEqual(context.device, "cpu")
        self.assertEqual(context.precision, "fp32")
        self.assertEqual(context.dtype, torch.float32)

    @patch(
        "unlearn_bench.utils.device.available_devices",
        return_value={"cuda": True, "mps": True, "cpu": True},
    )
    def test_auto_prefers_cuda(self, _available):
        self.assertEqual(_resolve_device("auto"), "cuda")

    @patch(
        "unlearn_bench.utils.device.available_devices",
        return_value={"cuda": False, "mps": True, "cpu": True},
    )
    def test_auto_prefers_mps_over_cpu(self, _available):
        self.assertEqual(_resolve_device("auto"), "mps")

    @patch(
        "unlearn_bench.utils.device.available_devices",
        return_value={"cuda": False, "mps": False, "cpu": True},
    )
    def test_unavailable_explicit_device_fails(self, _available):
        with self.assertRaises(DeviceUnavailableError):
            _resolve_device("cuda")

    def test_cpu_fp16_is_rejected(self):
        with self.assertRaises(PrecisionUnavailableError):
            select_device("cpu", "fp16")

    @patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    @patch("unlearn_bench.utils.device._resolve_precision", return_value=("fp32", torch.float32))
    @patch("unlearn_bench.utils.device._resolve_device", return_value="mps")
    def test_unapproved_mps_fallback_fails(self, _device, _precision):
        with self.assertRaisesRegex(RuntimeError, "unrecorded CPU fallback"):
            select_device("mps", "fp32", allow_mps_fallback=False)

    @patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    @patch("unlearn_bench.utils.device._resolve_precision", return_value=("fp32", torch.float32))
    @patch("unlearn_bench.utils.device._resolve_device", return_value="mps")
    def test_approved_mps_fallback_is_recorded(self, _device, _precision):
        context = select_device("mps", "fp32", allow_mps_fallback=True)
        self.assertTrue(context.mps_fallback_enabled)


if __name__ == "__main__":
    unittest.main()
