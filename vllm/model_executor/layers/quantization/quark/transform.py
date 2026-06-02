# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def build_hadamard_matrix(
    size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """Build Sylvester Hadamard matrix of size (size, size). size must be a power of 2.
    Values are +1 and -1. Used when Quark export does not include the Hadamard matrix."""
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError("Hadamard size must be a positive power of 2")
    h = torch.tensor([[1]], device=device, dtype=torch.int8)
    h2 = torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.int8)
    while h.shape[0] < size:
        h = torch.kron(h2, h)
    if h.shape[0] > size:
        raise ValueError("size must be power of 2")
    return h.to(dtype)


class OrthogonalTransform(torch.nn.Module):
    def __init__(
        self,
        input_rotation: torch.Tensor,
        rotation_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.input_rotation = input_rotation
        self.rotation_size = input_rotation.shape[0]
        self.rotation_config = rotation_config

    def forward(self, x: torch.Tensor):
        needs_reshape = False
        if x.shape[-1] != self.rotation_size:
            needs_reshape = True
            x = x.reshape(*x.shape[:-1], -1, self.rotation_size)

        x = x @ self.input_rotation

        if needs_reshape:
            x = x.reshape(*x.shape[:-2], -1)

        return x

    @staticmethod
    def setup_transform(quant_config: dict[str, Any], layer_names: list[str]):
        use_online_rotation = False
        rotation_config = None
        rotation_size = None

        if (
            quant_config.get("algo_config") is not None
            and len(quant_config["algo_config"]) > 0
            and quant_config["algo_config"][0]["name"] == "rotation"
        ):
            rotation_config = quant_config["algo_config"][0]

            online_rotation_layers = get_online_rotation_layers(rotation_config) or None

            if online_rotation_layers is not None and any(
                any(ol in layer_name for ol in online_rotation_layers) for layer_name in layer_names
            ):
                use_online_rotation = True
                rotation_size = rotation_config["rotation_size"]

                if rotation_size is None:
                    raise NotImplementedError("rotation_size=None is not supported")

        return use_online_rotation, rotation_config, rotation_size

    def post_process_transform(self):
        if self.rotation_config is not None and not self.rotation_config["trainable"]:
            # In case hadamard transform is used (non-trained case), it is
            # serialized as torch.int8 with only `-1` and `1` values.
            self.input_rotation.data = self.input_rotation.data.to(
                torch.float
            ) / math.sqrt(self.rotation_size)

        rotation_dtype = torch.get_default_dtype()
        self.input_rotation.data = self.input_rotation.data.to(rotation_dtype)

        logger.debug("self.rotation_size: %s", self.rotation_size)
        logger.debug("self.input_rotation.data: %s", self.input_rotation.data)


def get_online_rotation_layers(rotation_config: dict) -> list[str]:
    """Extract the list of online-rotation layer names from a rotation config dict.

    Supports two Quark formats:
      - New (≥0.12): online_config.online_rotation_layers  (explicit list)
      - Legacy (0.11 Hadamard): online_r1_rotation=True + scaling_layers  (inferred)

    Returns an empty list if online rotation is not configured.
    """
    online_cfg = rotation_config.get("online_config") or {}
    layers = online_cfg.get("online_rotation_layers")
    if layers:
        return list(layers)

    if rotation_config.get("online_r1_rotation"):
        scaling = rotation_config.get("scaling_layers", {})
        names: set[str] = set()
        for group in ("first_layer", "middle_layers", "last_layer"):
            for entry in scaling.get(group, []):
                for mod in entry.get("next_modules", []):
                    base = (mod
                            .replace("model.layers.layer_id.", "")
                            .replace("model.layers.pre_layer_id.", ""))
                    names.add(base)
        return list(names)

    return []


def rotation_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str | None = None,
    shard_id: str | None = None,
    expert_id: int | None = None,
):
    assert param.shape == loaded_weight.shape
    # Handle dtype conversion: Quark may export bool, vLLM expects int8/float64
    if param.dtype != loaded_weight.dtype:
        # Hadamard export uses bool; vLLM expects int8 with True->1, False->-1
        if loaded_weight.dtype == torch.bool and param.dtype == torch.int8:
            loaded_weight = torch.where(loaded_weight, 1, -1).to(torch.int8)
        else:
            loaded_weight = loaded_weight.to(param.dtype)
    param.data.copy_(loaded_weight)
