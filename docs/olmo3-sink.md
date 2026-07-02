# Olmo3Sink

Olmo3Sink is the OLMo3 training/inference path used for proof-reasoning experiments. It starts from the Hugging Face OLMo3 architecture and adds trainable attention sinks plus packed-sequence metadata reuse for FlashAttention backends.

## Code Layout

| Path | Purpose |
|---|---|
| `src/prime_rl/trainer/models/olmo3_sink/configuration_olmo3_sink.py` | `Olmo3SinkConfig`, registered as `model_type = "olmo3_sink"`. |
| `src/prime_rl/trainer/models/olmo3_sink/modeling_olmo3_sink.py` | Trainer-side model implementation with attention sinks and OLMo3 per-layer RoPE handling. |
| `src/prime_rl/trainer/models/olmo3_sink/vllm_adapter.py` | vLLM adapter with packed `qkv_proj`, packed `gate_up_proj`, and per-head sink loading. |
| `src/prime_rl/trainer/models/olmo3_sink/converting_olmo3_sink.py` | Layer conversion for vLLM kernel-format weight transfer, including optional FP8 quantized transfer. |

`Olmo3SinkConfig` and `Olmo3SinkForCausalLM` are registered in Prime-RL's custom model mapping, so `[trainer.model] impl = "custom"` loads the trainer-side implementation directly.

## Training Configuration

Use the custom trainer model implementation for Olmo3Sink:

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_3"
cp = 2
cp_style = "ulysses"
fp8 = false
```

Context parallelism (`cp`) is useful for long contexts on dense 32B models. The current 4xH200 smoke layout uses `cp = 2` over two trainer GPUs.

## FP8 Support

Trainer FP8 is enabled with:

```toml
[trainer.model]
fp8 = true
```

This uses Prime-RL's existing DeepGEMM blockwise FP8 path by replacing eligible `nn.Linear` modules with `Float8BlockwiseLinear`. It applies to dense OLMo3Sink projections such as attention and MLP linears. Layer norms, embeddings, LM head, and `self_attn.sinks` stay in their normal dtype.

For vLLM inference, use vLLM quantization:

```toml
[inference.vllm_extra]
quantization = "fp8"
```

If NCCL quantized weight transfer is enabled, Olmo3Sink emits vLLM adapter names directly:

- `self_attn.qkv_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_up_proj.weight`
- `mlp.down_proj.weight`
- matching `*.weight_scale_inv` tensors for FP8 weights

## Current 4xH200 OPD Layout

The current Modal test layout is:

| GPU | Role |
|---|---|
| 0 | policy vLLM rollout server |
| 1-2 | trainer with `cp = 2` |
| 3 | frozen OPD teacher vLLM server |

Known successful smoke:

- Dataset: `submissions-instructions/test.csv`
- Context: 2,048
- Trainer: 2 GPUs, `cp = 2`, `cp_style = "ulysses"`
- Policy vLLM: 1 GPU, FP8
- Teacher vLLM: 1 GPU, FP8
- Result: one trainer step completed, peak trainer memory about 99.7 GiB.

Current longer-context test command:

```bash
bash /workspace/submissions-instructions/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Default settings in that command:

- Dataset: `/workspace/submissions-instructions/imo_data_1959_2024.csv`
- Columns: `question` and `solution`
- Context length: 16,384
- Rollout max completion tokens: 12,288
- Batch size: 2
- Group size: 2
- Optimizer: Muon
- Trainer FP8: enabled
- Policy/teacher vLLM quantization: FP8

Override these from the shell when needed:

```bash
PRIME_OPD_CTX_LEN=16384 \
PRIME_OPD_COMPLETION_TOKENS=8192 \
MAX_TRAIN_STEPS=1 \
bash /workspace/submissions-instructions/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Practical Notes

- Keep `--fetch-update` enabled in submission-side commands so Modal and server runs pick up the latest `submissions-instructions` and `prime-rl` commits.
- For first-pass debugging, keep `wandb_mode = "disabled"` and run one step.
- Increase rollout context before increasing batch size. For proof data, long completions are usually the first pressure point.
- If vLLM memory is tight, lower `max_num_seqs`, `max_num_batched_tokens`, or `rollout_max_completion_tokens` before changing the trainer layout.
