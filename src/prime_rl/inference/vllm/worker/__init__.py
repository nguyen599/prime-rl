import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_fp32_lm_head,
    monkey_patch_fp32_router_logits,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
    monkey_patch_skip_deepseek_v4_sparse_mla_warmup,
    register_olmo3_sink_model,
)

logger = logging.getLogger(__name__)

# Register OLMo3Sink in worker-extension processes as well as API-server
# processes. This is idempotent and safe when no OLMo3Sink model is used.
register_olmo3_sink_model()
# Monkeypatch MiniMaxM2 MoE gate dtype and adapter key mapping for LoRA compatibility
monkey_patch_minimax_m2_for_lora()
# Disable LoRA on MoE layers so vLLM picks better kernels (e.g. TRTLLMFlashInfer on Blackwell)
if os.environ.get("PRIME_NO_MOE_LORA") == "1":
    logger.info("PRIME_NO_MOE_LORA=1: disabling LoRA on MoE layers")
    monkey_patch_no_moe_lora()
else:
    logger.info("PRIME_NO_MOE_LORA=0: no patch applied")

# Install fp32 lm_head patch; self-gates on additional_config["fp32_lm_head"] at call time
monkey_patch_fp32_lm_head()

# Install fp32 router logits patch; self-gates on additional_config["fp32_router_logits"]
monkey_patch_fp32_router_logits()
# Optional DeepSeek-V4 startup warmup skip. This worker module is imported in
# each vLLM worker-extension process before model warmup, so patching here is
# more reliable than relying on a sitecustomize shim.
monkey_patch_skip_deepseek_v4_sparse_mla_warmup()
