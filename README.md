<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Custom Modifications: Per-Token Speculative Decoding Trace Instrumentation

This fork (`trace-step1` branch) adds **per-token trace instrumentation** to vLLM's speculative decoding pipeline for research on spatial structure in acceptance/rejection patterns. The instrumentation is gated behind an environment variable — when disabled, there is zero performance overhead.

### Overview

When `VLLM_SPEC_DECODE_TRACE_FILE=/path/to/output.jsonl` is set, every completed request writes one JSONL line containing a metadata envelope and a `trace` array with **16 fields per generated token**. This enables analysis of where and why the target model accepts or rejects draft tokens.

### Modified Files (8 commits, 546 lines added)

#### 1. `vllm/v1/spec_decode/trace_state.py` — Side-Channel Data Store

**What changed:** Added two new module-level dictionaries and their accessor functions for passing trace data from GPU workers to the scheduler.

| Addition | Purpose |
|----------|---------|
| `_draft_trace_data` dict + `add_draft_trace()` / `pop_draft_trace()` | Stores per-token draft-side statistics (raw logits top-10, softmax top-10, entropy, top-1/top-5 probabilities) indexed by request ID |
| `_target_trace_data` dict + `add_target_trace()` / `pop_target_trace()` | Stores per-token target-side statistics (target probability of draft token, target top-1, KL divergence) indexed by request ID |
| `cleanup_request()` | Removes all trace state for a finished request to prevent memory leaks |

**Data flow:** GPU model runner → `trace_state` (module-level dicts) → scheduler reads at output time.

#### 2. `vllm/v1/spec_decode/eagle.py` — Draft Model Trace Capture

**What changed:** Extended `_greedy_sample()` method in `SpecDecodeBaseProposer` to compute and buffer additional draft statistics when tracing is enabled.

**Location:** `_greedy_sample()` method, inside the existing `if self._trace_draft_logprobs:` block (line ~414).

**New computations (all gated behind existing `_trace_draft_logprobs` flag):**
- `raw_top10 = torch.topk(logits_f, k=10)` — raw logits top-10 (before softmax)
- `probs = torch.softmax(logits_f, dim=-1)` — full-vocabulary softmax
- `sm_top10 = torch.topk(probs, k=10)` — softmax probabilities top-10
- `entropy = -(probs * torch.log2(probs + eps)).sum(dim=-1)` — Shannon entropy in **bits** (log base 2)
- `top1_prob = probs.max(dim=-1).values` — draft's top-1 probability
- `top5_prob = torch.topk(probs, k=5).values.sum(dim=-1)` — cumulative top-5 probability

**New buffer:** `self._draft_trace_buf` (list of dicts per draft step) + `pop_draft_trace_buf()` accessor.

#### 3. `vllm/v1/sample/rejection_sampler.py` — Target-Side Trace Extraction

**What changed:** Added `_extract_target_trace()` method to the `RejectionSampler` class, called inside `forward()` when `_TRACE_ENABLED` is True.

**New module-level flag:** `_TRACE_ENABLED = bool(os.environ.get("VLLM_SPEC_DECODE_TRACE_FILE"))` — read once at import time for zero overhead.

**`_extract_target_trace()` computes per draft position:**
- `target_prob_of_draft_token`: target model's softmax probability at the specific token the draft model chose
- `target_top1_token_id` / `target_top1_prob`: target model's argmax prediction
- `kl_divergence`: KL(draft || target) using top-20 approximation. Set to `None` when `draft_probs` is `None` (e.g., EAGLE greedy path), in which case the scheduler computes a fallback KL from draft softmax top-10 and target logprobs.

**Buffer:** `self._target_trace_buf` + `pop_target_trace_buf()`.

#### 4. `vllm/v1/worker/gpu_model_runner.py` — Data Deposit into Side-Channel

**What changed:** Two new code blocks deposit trace data from GPU workers into the `trace_state` module-level dicts.

**After rejection sampling (line ~3339):** Pops `rejection_sampler.pop_target_trace_buf()`, iterates over requests, deposits each position's target-side data via `trace_state.add_target_trace(req_id, pos_data)`.

**After draft proposal (line ~4749):** Pops `drafter.pop_draft_trace_buf()`, transposes from per-step batch tensors to per-request per-position dicts, deposits via `trace_state.add_draft_trace(req_id, rec)`. Each record contains: `logits_top10_ids`, `logits_top10_vals`, `softmax_top10_ids`, `softmax_top10_vals`, `entropy`, `top1_prob`, `top5_prob`.

#### 5. `vllm/v1/core/sched/scheduler.py` — Trace Assembly and JSONL Output (largest change: 319 lines)

**What changed:** Major restructuring of the trace accumulation and output logic.

**New state tracking (added to `__init__`):**
- `_spec_decode_trace_records`: per-request list of per-token trace dicts (all 16 fields)
- `_spec_decode_round_counter`: tracks which speculative round each token belongs to
- `_spec_decode_position_counter`: global token position counter per request
- `_spec_decode_start_time`: wall-clock timing per request
- `_trace_config`: experiment metadata loaded once from env vars or JSON config file

**New static methods:**
- `_load_trace_config()`: reads experiment metadata from `SPEC_DECODE_TRACE_CONFIG` JSON file and/or individual `SPEC_DECODE_TRACE_*` env vars (pair_id, dataset_id, quant_combo_id, etc.)
- `_approx_kl_from_topk()`: fallback KL(draft || target) approximation for the EAGLE path, using draft softmax top-10 and target logprobs

**Trace accumulation (in `_make_engine_core_outputs`):** For each speculative step, pops draft and target trace data from `trace_state`, builds a per-token record with all 16 fields. For non-speculative tokens (prefill, AR decode), all draft/target fields are set to `None`.

**JSONL output (in `_free_request`):** Replaced the old flat format:
```json
{"request_id": "...", "acceptance": [...], "target_logprobs": [...], "draft_logprobs": [...]}
```
With the nested metadata envelope + trace array:
```json
{
  "pair_id": "P1", "quant_combo_id": "C1", "dataset_id": "D1",
  "gamma": 5, "temperature": 0.0, "acceptance_rate": 0.85,
  "total_accepted": 200, "total_rejected": 35,
  "num_speculative_rounds": 47, "wall_time_seconds": 1.23,
  "trace": [
    {"position": 0, "token_id": 123, "accepted": true, "draft_entropy": 3.2, ...},
    ...
  ]
}
```

#### 6. `vllm/config/speculative.py` — Vocab Size Mismatch Tolerance

**What changed:** Relaxed `verify_equal_vocab_size_if_draft_model()` to allow `draft_vocab_size < target_vocab_size`.

**Reason:** Qwen2.5 models share the same tokenizer but pad vocabulary to different multiples of 128 for tensor parallelism alignment (0.5B: 151,936 tokens vs 7B: 152,064 tokens). The extra 128 entries in the larger model are unreachable dummy padding tokens. The original code raised a hard `ValueError` on any mismatch; now it logs a warning when `draft < target` and only errors when `draft > target`.

### Per-Token Trace Fields (16 fields)

| # | Field | Type | Source |
|---|-------|------|--------|
| 1 | `position` | int | scheduler position counter |
| 2 | `token_id` | int | generated token ID |
| 3 | `token_str` | str/null | null (post-processing decodes) |
| 4 | `accepted` | bool | scheduler acceptance mask |
| 5 | `is_rejection_position` | bool | scheduler (first rejected in round) |
| 6 | `draft_logits_top10` | list[{id,logit}] | eagle.py raw logits |
| 7 | `draft_softmax_top10` | list[{id,prob}] | eagle.py softmax |
| 8 | `draft_entropy` | float (bits) | eagle.py Shannon entropy (log2) |
| 9 | `draft_top1_prob` | float | eagle.py max(softmax) |
| 10 | `draft_top5_prob` | float | eagle.py sum(top-5 softmax) |
| 11 | `target_prob_of_draft_token` | float | rejection_sampler.py |
| 12 | `target_top1_token_id` | int | rejection_sampler.py |
| 13 | `target_top1_prob` | float | rejection_sampler.py |
| 14 | `kl_divergence` | float | rejection_sampler.py or scheduler fallback |
| 15 | `speculative_round` | int | scheduler round counter |
| 16 | `position_in_round` | int | scheduler (0 to gamma-1) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `VLLM_SPEC_DECODE_TRACE_FILE` | Output JSONL path. When unset, all tracing is disabled (zero overhead). |
| `SPEC_DECODE_TRACE_CONFIG` | Path to JSON file with experiment metadata (pair_id, dataset_id, etc.) |
| `SPEC_DECODE_TRACE_PAIR_ID` | Override pair_id in trace output |
| `SPEC_DECODE_TRACE_DATASET_ID` | Override dataset_id in trace output |
| `SPEC_DECODE_TRACE_QUANT_COMBO_ID` | Override quant_combo_id in trace output |
| `SPEC_DECODE_TRACE_DRAFT_MODEL` | Override draft_model name |
| `SPEC_DECODE_TRACE_TARGET_MODEL` | Override target_model name |
| `SPEC_DECODE_TRACE_DRAFT_QUANT` | Override draft_quant |
| `SPEC_DECODE_TRACE_TARGET_QUANT` | Override target_quant |
| `SPEC_DECODE_TRACE_GAMMA` | Override gamma |
| `SPEC_DECODE_TRACE_TEMPERATURE` | Override temperature |
| `SPEC_DECODE_TRACE_SAMPLE_ID` | Override sample_id |

### Usage Example

```bash
export VLLM_SPEC_DECODE_TRACE_FILE=/path/to/traces.jsonl
export SPEC_DECODE_TRACE_PAIR_ID=P1
export SPEC_DECODE_TRACE_DATASET_ID=D1

python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='target_model', speculative_config={'model': 'draft_model', 'num_speculative_tokens': 5})
outputs = llm.generate(['Hello world'], SamplingParams(temperature=0.0, max_tokens=50))
"
# traces.jsonl now contains one line per completed request with all 16 per-token fields
```

### Design Decisions

1. **Zero overhead when disabled:** All computations (softmax, entropy, top-k extraction, KL divergence) are inside `if _trace_enabled:` checks. When `VLLM_SPEC_DECODE_TRACE_FILE` is unset, no extra GPU kernels, no extra tensor operations.

2. **`token_str` is null:** The scheduler has no tokenizer access. A post-processing script decodes token IDs using the model's tokenizer.

3. **KL divergence dual path:** The rejection sampler computes exact KL when `draft_probs` is available. For EAGLE (where `draft_probs=None`), the scheduler computes an approximate KL from draft softmax top-10 and target logprobs.

4. **Side-channel architecture:** Draft trace data flows through `trace_state.py` module-level dicts (GPU model runner → scheduler) rather than being threaded through function signatures, to minimize changes to existing APIs.

---

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
