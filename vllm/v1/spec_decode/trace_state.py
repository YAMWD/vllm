# SPDX-License-Identifier: Apache-2.0
"""
Shared in-process state for speculative decoding acceptance tracing.

Both the GPU model runner and the scheduler run in the same EngineCore
subprocess. This module-level dict lets the model runner deposit per-request
draft logprobs that the scheduler reads when writing the trace file.

Layout of _draft_logprobs[req_id]:
  A flat list of dicts, one per *draft token position* proposed across all
  spec-decode rounds for this request.  Each dict maps token_id -> logprob
  (top-K from the draft model's distribution at that position).

The scheduler knows num_accepted per round and uses it to align the flat list
with the per-output-token acceptance mask.

Extended trace data (_draft_trace_data[req_id]):
  A flat list of dicts, one per *draft token position*, containing richer
  per-position statistics: raw logits top-10, softmax top-10, entropy,
  top-1/top-5 probabilities.

Target trace data (_target_trace_data[req_id]):
  A flat list of dicts deposited from the rejection sampler, containing
  per-position target-side statistics: target_prob_of_draft_token,
  target_top1_token_id, target_top1_prob, kl_divergence, plus (Mod A)
  target_top10_logits and target_top10_softmax for symmetric KL
  computation on the union of draft + target top-10 ids.
"""

from __future__ import annotations

from typing import Any

# req_id -> list[ {token_id: logprob} ] (one entry per draft token position)
_draft_logprobs: dict[str, list[dict[int, float]]] = {}

# req_id -> list[ dict with keys:
#   "logits_top10_ids": list[int],
#   "logits_top10_vals": list[float],
#   "softmax_top10_ids": list[int],
#   "softmax_top10_vals": list[float],
#   "entropy": float,
#   "top1_prob": float,
#   "top5_prob": float,
# ]
_draft_trace_data: dict[str, list[dict[str, Any]]] = {}

# req_id -> list[ dict with keys:
#   "target_prob_of_draft_token": float,
#   "target_top1_token_id": int,
#   "target_top1_prob": float,
#   "target_top10_logits": list[{"id": int, "logit": float}],   # Mod A
#   "target_top10_softmax": list[{"id": int, "prob": float}],   # Mod A
#   "kl_divergence": float | None,
# ]
_target_trace_data: dict[str, list[dict[str, Any]]] = {}


def add_draft_logprobs(req_id: str,
                       logprobs: dict[int, float]) -> None:
    """Append one draft-position logprob dict for a request."""
    _draft_logprobs.setdefault(req_id, []).append(logprobs)


def pop_draft_logprobs(req_id: str) -> list[dict[int, float]]:
    """Remove and return all accumulated draft logprobs for a request."""
    return _draft_logprobs.pop(req_id, [])


def add_draft_trace(req_id: str, data: dict[str, Any]) -> None:
    """Append one draft-position trace record for a request."""
    _draft_trace_data.setdefault(req_id, []).append(data)


def pop_draft_trace(req_id: str) -> list[dict[str, Any]]:
    """Remove and return all accumulated draft trace data for a request."""
    return _draft_trace_data.pop(req_id, [])


def add_target_trace(req_id: str, data: dict[str, Any]) -> None:
    """Append one target-position trace record for a request."""
    _target_trace_data.setdefault(req_id, []).append(data)


def pop_target_trace(req_id: str) -> list[dict[str, Any]]:
    """Remove and return all accumulated target trace data for a request."""
    return _target_trace_data.pop(req_id, [])


def cleanup_request(req_id: str) -> None:
    """Remove all trace state for a finished request."""
    _draft_logprobs.pop(req_id, None)
    _draft_trace_data.pop(req_id, None)
    _target_trace_data.pop(req_id, None)
