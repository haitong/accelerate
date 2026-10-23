# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the TorchTPU (`torch_tpu`) integration.

TorchTPU exposes TPUs as a `PrivateUse1` backend renamed to `tpu` and is driven exclusively through `torch.accelerator`.
These tests run on CPU-only CI by registering the `tpu` device name and mocking the handful of `torch.accelerator` and
`torch.distributed` calls that `accelerate` makes, so they verify the plumbing (device discovery, backend selection and
device binding) rather than the TPU runtime itself. The hardware path is exercised by `@require_torch_tpu` tests.
"""

import os
import unittest
from unittest import mock

import torch

from accelerate.commands.config.config_utils import _convert_distributed_mode
from accelerate.state import PartialState
from accelerate.utils import DistributedType, DynamoBackend, RNGType, is_torch_tpu_available
from accelerate.utils.environment import get_current_device_type
from accelerate.utils.memory import clear_device_cache


def _register_fake_tpu_device():
    """Makes `torch.device("tpu")` constructible without `torch_tpu` installed (no-op if TorchTPU already did it)."""
    try:
        torch.utils.rename_privateuse1_backend("tpu")
    except RuntimeError:
        # Already renamed (to "tpu" by TorchTPU itself, or by a previous call); nothing to do.
        pass


_HAS_TORCH_ACCELERATOR = hasattr(torch, "accelerator")


@unittest.skipUnless(_HAS_TORCH_ACCELERATOR, "TorchTPU support relies on `torch.accelerator`")
class TorchTPUStateTester(unittest.TestCase):
    def setUp(self):
        _register_fake_tpu_device()
        PartialState._reset_state()

    def tearDown(self):
        PartialState._reset_state()

    def test_enums(self):
        assert DistributedType("MULTI_TPU") is DistributedType.MULTI_TPU
        assert RNGType("tpu") is RNGType.TPU
        assert DynamoBackend("TPU") is DynamoBackend.TPU
        # `accelerate config` menu order: "multi-TPU" is inserted right before "TPU" (PyTorch/XLA).
        assert _convert_distributed_mode(10) is DistributedType.MULTI_TPU
        assert _convert_distributed_mode(11) is DistributedType.XLA

    def test_is_torch_tpu_available_without_torch_tpu(self):
        # Must be a cheap, exception-free `False` on machines without the backend installed.
        is_torch_tpu_available.cache_clear()
        with mock.patch("importlib.util.find_spec", return_value=None):
            assert is_torch_tpu_available() is False
            assert is_torch_tpu_available(check_device=True) is False
        is_torch_tpu_available.cache_clear()

    def test_single_process_defaults_to_tpu_device(self):
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "RANK", "WORLD_SIZE")}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("accelerate.state.is_torch_tpu_available", return_value=True),
            mock.patch("accelerate.state.is_mps_available", return_value=False),
        ):
            state = PartialState()
            assert state.distributed_type == DistributedType.NO
            assert state.device.type == "tpu"
            assert state.num_processes == 1

    def test_multi_tpu_backend_and_device_binding(self):
        env = {"LOCAL_RANK": "1", "RANK": "1", "WORLD_SIZE": "4", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29999"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("accelerate.state.is_torch_tpu_available", return_value=True),
            mock.patch("accelerate.state.is_mps_available", return_value=False),
            mock.patch.object(torch.accelerator, "device_count", return_value=4),
            mock.patch.object(torch.accelerator, "set_device_index") as set_device_index,
            mock.patch.object(torch.distributed, "is_initialized", return_value=False),
            mock.patch.object(torch.distributed, "init_process_group") as init_process_group,
            mock.patch.object(torch.distributed, "get_world_size", return_value=4),
            mock.patch.object(torch.distributed, "get_rank", return_value=1),
        ):
            state = PartialState()

            assert state.distributed_type == DistributedType.MULTI_TPU
            # TorchTPU's c10d backend; collectives are lowered to XLA, there is no NCCL-like library to choose.
            assert state.backend == "tpu_dist"
            init_process_group.assert_called_once()
            assert init_process_group.call_args.kwargs["backend"] == "tpu_dist"

            # One process per TPU device: `LOCAL_RANK` selects the device through `torch.accelerator`, never
            # through the (internal) `torch.tpu.set_device`.
            assert state.device == torch.device("tpu", 1)
            set_device_index.assert_called_once_with(torch.device("tpu", 1))
            assert state.num_processes == 4
            assert state.process_index == 1
            assert state.local_process_index == 1

    def test_multi_tpu_rejects_mismatched_backend(self):
        env = {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "2"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("accelerate.state.is_torch_tpu_available", return_value=True),
            mock.patch("accelerate.state.is_mps_available", return_value=False),
            self.assertRaises(ValueError),
        ):
            PartialState(backend="nccl")

    def test_get_current_device_type(self):
        with mock.patch("accelerate.utils.imports.is_torch_tpu_available", return_value=True):
            assert get_current_device_type() == ("tpu", "MULTI_TPU")

    def test_clear_device_cache_uses_torch_accelerator(self):
        with (
            mock.patch("accelerate.utils.memory.is_torch_tpu_available", return_value=True),
            mock.patch("accelerate.utils.memory.is_cuda_available", return_value=False),
            mock.patch("accelerate.utils.memory.is_mps_available", return_value=False),
            mock.patch.object(torch.accelerator, "empty_cache") as empty_cache,
        ):
            clear_device_cache()
            empty_cache.assert_called_once()
