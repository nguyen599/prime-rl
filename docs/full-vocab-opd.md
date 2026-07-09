# Full-Vocab OPD

Full-vocab OPD is an opt-in variant of `opd` that distills against the
teacher's full next-token distribution instead of only the sampled token
logprob.

The default `opd` path is unchanged. If you do not set
`distill_mode = "full_vocab_hidden"`, Prime-RL still asks the teacher for
`prompt_logprobs`, ships `ref_logprobs`, and trains the existing scalar
reverse-KL signal.

## Why Use It

Standard OPD only compares teacher and student on the token that the policy
sampled. That is cheap, but it discards the teacher's distribution over the rest
of the vocabulary.

Full-vocab OPD sends the teacher's last hidden states to the trainer. The trainer
loads the teacher LM head, reconstructs teacher logits, computes the student's
logits with the live student LM head, and applies chunked reverse KL over the
full vocabulary:

```text
KL(P_student || P_teacher)
```

This matches the "teacher hidden states plus LM head reconstruction" style used
by full-vocab OPD pipelines while avoiding materializing a full `[tokens, vocab]`
tensor at once.

The student and teacher hidden sizes do not need to match. The student hidden is
projected by the student LM head `[vocab, student_hidden]`, and the teacher
hidden is projected by the teacher LM head `[vocab, teacher_hidden]`. What must
match is the vocabulary dimension and token-id semantics, otherwise the KL would
compare different tokens.

## Configuration

Enable it under `[orchestrator.algo]` and `[trainer.full_vocab_distill]`:

```toml
[orchestrator.algo]
type = "opd"
distill_mode = "full_vocab_hidden"
teacher_hidden_dtype = "bfloat16"
teacher_hidden_transport = "filesystem"
teacher_hidden_path = "/shared/prime-rl/teacher-hidden/run-001"

[orchestrator.algo.teacher]
name = "/models/teacher"
base_url = ["http://localhost:8001/v1"]
skip_model_check = true

[trainer.full_vocab_distill]
enabled = true
teacher_lm_head_path = "/models/teacher"
token_chunk_size = 64
vocab_chunk_size = 8192
teacher_hidden_dtype = "bfloat16"
```

If `teacher_lm_head_key` is not set, the trainer tries common HF keys in this
order:

```text
lm_head.weight
head.weight
model.embed_tokens.weight
model.embedding.weight
transformer.wte.weight
```

Set `teacher_lm_head_key` when using a checkpoint with a non-standard key.

## Data Flow

1. The policy generates rollouts as usual.
2. `OPDAlgorithm.score_rollout()` checks `distill_mode`.
3. In default mode, it calls `InferencePool.score()` and fills
   `sample.ref_logprobs`.
4. In full-vocab inline mode, it fills `sample.ref_hidden_states`. In filesystem
   mode, the teacher atomically writes a self-describing tensor file and the
   orchestrator receives only `sample.ref_hidden_states_file` metadata.
5. The packer truncates file references by changing metadata only. The
   filesystem microbatch sender creates a private hard link for each owning
   trainer rank, then removes the producer name.
6. The trainer loads the teacher LM head once at startup.
7. The owning trainer rank memory-maps its file segments, concatenates them in
   packed-token order, fills only padding rows with zeros, and unlinks its
   private handles.
8. During the model forward, `FusedOutputLinear` computes normal sampled-token
   logprobs and the optional full-vocab KL loss.
9. The full-vocab KL is normalized by the global `ref_kl` token count, matching
   the existing component-normalization behavior.

## vLLM Endpoint

Prime-RL adds a custom worker route:

```text
POST /prime_rl/prefill_hidden_states
```

The route calls a worker RPC named `prefill_hidden_states`. The legacy inline
response is an encoded tensor:

```json
{
  "dtype": "bfloat16",
  "shape": [seq_len, hidden_size],
  "data": "<base64 raw tensor bytes>"
}
```

This endpoint is only used by `distill_mode = "full_vocab_hidden"`.

With `teacher_hidden_transport = "filesystem"`, the request includes a unique
absolute output path and the response is only a handle:

```json
{
  "transport": "filesystem",
  "path": "/shared/prime-rl/teacher-hidden/run-001/....prlhs",
  "dtype": "bfloat16",
  "shape": [40000, 4096],
  "offset": 64,
  "nbytes": 327680000
}
```

With the default hook backend, the payload never enters the API server,
orchestrator, training-batch msgpack, or packer process. The path must resolve
to the same shared filesystem from all roles. The default remains `inline` for
backward compatibility.

## Current Limits

Full-vocab OPD currently has conservative guards:

- Teacher hidden-state scoring supports vLLM tensor parallelism when the
  teacher model runner returns replicated final hidden states after TP
  collectives. The API server queries all workers and uses the first non-null
  response.
- Trainer context parallelism must be CP=1.
- Trainer must use an integer `model.fused_lm_head_token_chunk_size`.
- The teacher LM-head tensor must be available as HF safetensors.
- The student and teacher vocabularies must be aligned. Hidden sizes may differ
  as long as each hidden state matches its own LM head.
- Hidden states are stored as float16, bfloat16, or float32 raw tensor bytes;
  bfloat16 is the default. Filesystem transport removes the control-plane copy,
  but there is no int6 hidden-state compression yet.

These limits are intentional. They prevent silent wrong KL when trainer hidden
states or LM-head weights are sharded in a layout the trainer does not yet
reconstruct. If a new vLLM backend returns sharded final hidden states instead
of replicated hidden states, add an explicit gather before using it for
full-vocab OPD.

## Performance Knobs

`token_chunk_size` controls how many selected training tokens are processed per
full-vocab KL chunk. Smaller values reduce memory; larger values may improve
throughput.

`vocab_chunk_size` controls vocabulary chunking. Use smaller values if the
LM-head pass OOMs, and larger values if there is memory headroom.

Only tokens with nonzero `ref_kl_weights` enter the full-vocab pass. Prompt,
padding, CE-only, and masked tokens are skipped.

For long-context or high-concurrency runs, use filesystem transport. Inline
transport base64-encodes the full tensor over HTTP and then copies it through
the orchestrator and packer, which is suitable only for compatibility and
small smoke tests.

Filesystem producer files are atomically written (`tmp` + rename). The
filesystem microbatch sender hard-links each segment into the owning rank's
step directory; the trainer removes that private link after mmap. Producer
files left by crashes are swept after `PRIME_RL_HIDDEN_STATE_TTL_SECONDS`
(default 21600 seconds), at most once per
`PRIME_RL_HIDDEN_STATE_SWEEP_INTERVAL_SECONDS` (default 600 seconds).

## Troubleshooting

`hidden-state scorer returned no result`
: The teacher endpoint did not return a worker hidden-state tensor. Check that
  the vLLM worker patch is loaded and that `/prime_rl/prefill_hidden_states`
  is registered on the teacher server.

`full-vocab OPD distillation currently requires trainer context parallel size 1`
: CP sharding is not supported for this path yet. Set trainer CP to 1.

`could not find teacher LM-head tensor`
: Set `teacher_lm_head_path` to the teacher HF checkpoint directory and, if
  needed, set `teacher_lm_head_key` to the exact tensor key.

`full-vocab OPD distillation requires ref_hidden_states`
: The orchestrator was not running in `distill_mode = "full_vocab_hidden"`, or
  the teacher endpoint did not expose `/prime_rl/prefill_hidden_states`.

`teacher_hidden_path is required for filesystem hidden-state transport`
: Set one absolute directory that is mounted at the same path on the teacher
  and trainer nodes. Do not use node-local `/tmp` unless the cluster mounts it
  as shared storage.

## Backward Compatibility

The legacy path remains the default:

```toml
[orchestrator.algo]
type = "opd"

[orchestrator.algo.teacher]
name = "/models/teacher"
base_url = ["http://localhost:8001/v1"]
```

This emits no `trainer.full_vocab_distill` section and uses `ref_logprobs` as
before.
