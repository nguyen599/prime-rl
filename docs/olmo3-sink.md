# Olmo3Sink

Olmo3Sink is the OLMo3 training/inference path used for proof-reasoning experiments. It starts from the Hugging Face OLMo3 architecture and adds trainable attention sinks plus packed-sequence metadata reuse for FlashAttention backends.

## Code Layout

| Path | Purpose |
|---|---|
| `src/prime_rl/trainer/models/olmo3_sink/configuration_olmo3_sink.py` | `Olmo3SinkConfig`, registered as `model_type = "olmo3_sink"`. |
| `src/prime_rl/trainer/models/olmo3_sink/modeling_olmo3_sink.py` | Trainer-side model implementation with attention sinks and OLMo3 per-layer RoPE handling. |
| `src/prime_rl/trainer/models/olmo3_sink/magi_sink.py` | Lazy FA2/FA3/FA4 dispatcher for MagiAttention's sink extensions. |
| `src/prime_rl/trainer/models/olmo3_sink/vllm_adapter.py` | vLLM adapter with packed `qkv_proj`, packed `gate_up_proj`, and per-head sink loading. |
| `src/prime_rl/trainer/models/olmo3_sink/converting_olmo3_sink.py` | Layer conversion for vLLM kernel-format weight transfer, including optional FP8 quantized transfer. |

`Olmo3SinkConfig` and `Olmo3SinkForCausalLM` are registered in Prime-RL's custom model mapping, so `[trainer.model] impl = "custom"` loads the trainer-side implementation directly.

## Training Configuration

Use the custom trainer model implementation for Olmo3Sink:

```toml
[trainer.model]
impl = "custom"
attn = "olmo3_sink_fa2"
cp = 2
cp_style = "ulysses"
fp8 = false
```

Context parallelism (`cp`) is useful for long contexts on dense 32B models. The current 4xH200 smoke layout uses `cp = 2` over two trainer GPUs.

### Sink Backends

| Prime-RL name | Intended hardware | Sliding window | Notes |
|---|---|---|---|
| `olmo3_sink_fa2` | Supported by the installed FA2 build | Yes | Default for standard OLMo3 mixed full/sliding layers. |
| `olmo3_sink_fa3` | Hopper (SM90) | Yes | Uses Magi's FA3 sink extension. |
| `olmo3_sink_fa4` | Blackwell (SM100+) | No | Only valid when every model layer uses full attention. |

All three paths preserve gradients to `self_attn.sinks`. Generic `flash_attention_*`
names are rejected for Olmo3Sink because those interfaces can silently drop `s_aux`.
Context parallel runs must use `cp_style = "ulysses"`; ring attention is not sink-aware.

### Magi Installation

The sink adapters need Magi's Python/Triton correction utilities, not its distributed
attention CUDA/communication extensions. A minimal install can therefore skip Magi's
CUDA build and reuse the FlashAttention packages already present in the image:

```bash
git clone https://github.com/SandAI-org/MagiAttention.git
cd MagiAttention
MAGI_ATTENTION_SKIP_CUDA_BUILD=1 \
  pip install --no-build-isolation --no-deps .
pip install --no-build-isolation --no-deps ./extensions
```

MagiAttention `efaabdbc` currently imports its optional DSA test helper from the
extensions package, so `expecttest` must also be installed. The production image pins
the Magi commit and installs that small dependency explicitly.

Run a single-GPU kernel canary with:

```bash
python tests/manual/olmo3_sink_magi_kernel.py --backend olmo3_sink_fa2
python tests/manual/olmo3_sink_magi_kernel.py --backend olmo3_sink_fa3
```

Run the CP parity canary with:

```bash
OLMO3_SINK_ATTN=olmo3_sink_fa2 torchrun --standalone --nproc-per-node=2 \
  tests/manual/olmo3_sink_cp_parity.py
```

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

If quantized weight transfer is enabled, Olmo3Sink emits vLLM adapter names directly:

- `self_attn.qkv_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_up_proj.weight`
- `mlp.down_proj.weight`
- matching scalar `*.weight_scale` tensors for FP8 weights

The policy launcher uses vLLM's online per-tensor `quantization = "fp8"`
method. During initial loading vLLM quantizes each HF `[N, K]` linear weight,
stores the live kernel parameter as `[K, N]`, and keeps one scalar scale. The
kernel-format exporter reproduces that live layout directly; it does not use
the trainer's blockwise FP8 layout for policy weight transfer.

The same prepacked tensors can be transferred through NCCL or a shared filesystem:

```toml
[weight_broadcast]
type = "filesystem"
quantize_in_weight_transfer = true
```

For filesystem transfer, the trainer converts each layer once, writes FP8
kernel-format safetensor shards, and publishes a manifest before the `STABLE`
marker. Each policy worker detects that manifest and copies the tensors into the
existing vLLM model in place. This avoids vLLM's checkpoint-format layerwise
reload and per-worker FP8 re-quantization. With the option disabled, filesystem
updates retain the original Hugging Face checkpoint reload behavior.

This path requires `trainer.model.impl = "custom"`, a model implementation with
`convert_layer_to_vllm_kernel`, and an FP8 vLLM policy model. It is intended for
the current Olmo3Sink policy layout with inference TP=1; tensor-parallel policy
workers require model-specific sharding support in the kernel loader.

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
