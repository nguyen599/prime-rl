import shutil
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers, preprocess_layer_quantized
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.transport.kernel_weights import save_kernel_weight_manifest, save_kernel_weight_shard
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.quantize_in_weight_transfer = config.quantize_in_weight_transfer
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.logger.debug(
            "Filesystem broadcast initialized "
            f"(save_format={config.save_format}, save_sharded={self.save_sharded}, "
            f"quantize_in_weight_transfer={self.quantize_in_weight_transfer})"
        )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        adapter_only = self.lora_config is not None

        if self.quantize_in_weight_transfer:
            if adapter_only:
                raise ValueError("Kernel-format filesystem weight transfer does not support LoRA adapters")
            self._broadcast_kernel_weights(model)
            if self.world.is_master:
                self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")
            return

        if not adapter_only:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_to_hf(state_dict)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                state_dict = revert_weight_conversion(model, state_dict)

        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.logger.debug(
                f"Broadcasting weights for run {idx} (ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            if adapter_only:
                # For adapter-only, MultiRunManager creates state dict directly for each run
                # All ranks must participate in DTensor gathering, but only master saves
                state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
                for key, value in state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if self.world.is_master:
                        state_dict[key] = value.to("cpu", non_blocking=False)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    # pack() already advanced progress to the next step, so the model we just
                    # trained — policy v(step-1) — broadcasts to broadcasts/step_{step-1}.
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step - 1,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    self.logger.debug(f"Saving weights for run {idx} to {save_dir}")
                    save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
                    if adapter_only:
                        orch_lora = self.multi_run_manager.config[idx].model.lora
                        save_lora_config(
                            model,
                            save_dir,
                            rank=orch_lora.rank,
                            alpha=orch_lora.alpha,
                            dropout=self.lora_config.dropout,
                        )

                    self._notify_orchestrator(save_dir)

                    # If the run is deleted, remove the run directory
                    # This is avoid the creation of zombie runs when the directory is deleted while we are broadcasting which recreates the directory
                    if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                        shutil.rmtree(self.multi_run_manager.get_run_dir(idx))

                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    @torch.no_grad()
    def _broadcast_kernel_weights(self, model: nn.Module) -> None:
        if not isinstance(model, PreTrainedModelPrimeRL):
            raise ValueError("Kernel-format filesystem weight transfer requires a custom Prime-RL trainer model")

        ready_idxs = list(self.multi_run_manager.ready_to_update_idxs)
        save_dirs: dict[int, Path] = {}
        if self.world.is_master:
            for idx in ready_idxs:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step - 1,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                save_dirs[idx] = save_dir
                self.logger.debug(f"Saving FP8 kernel weights for run {idx} to {save_dir}")

        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        filenames: list[str] = []

        for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            for key, value in list(layer_state_dict.items()):
                if isinstance(value, DTensor):
                    layer_state_dict[key] = value.to(torch.bfloat16).full_tensor()

            if not self.world.is_master:
                continue

            kernel_state = preprocess_layer_quantized(model, layer_state_dict, layer_idx)
            filename = "kernel-non-layer.safetensors" if layer_idx < 0 else f"kernel-layer-{layer_idx:05d}.safetensors"
            if kernel_state:
                filenames.append(filename)
                for save_dir in save_dirs.values():
                    save_kernel_weight_shard(save_dir, filename, kernel_state)
            del kernel_state

        if self.world.is_master:
            for idx, save_dir in save_dirs.items():
                save_kernel_weight_manifest(save_dir, filenames, quantized_fp8=True)
                self._notify_orchestrator(save_dir)
                self.multi_run_manager.ready_to_update[idx] = False

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, interval_to_keep: int | None):
        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step - 1,
                interval_to_keep,
            )
