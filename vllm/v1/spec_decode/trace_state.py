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
"""

from __future__ import annotations

# req_id -> list[ {token_id: logprob} ] (one entry per draft token position)
_draft_logprobs: dict[str, list[dict[int, float]]] = {}


def add_draft_logprobs(req_id: str,
                       logprobs: dict[int, float]) -> None:
    """Append one draft-position logprob dict for a request."""
    _draft_logprobs.setdefault(req_id, []).append(logprobs)


def pop_draft_logprobs(req_id: str) -> list[dict[int, float]]:
    """Remove and return all accumulated draft logprobs for a request."""
    return _draft_logprobs.pop(req_id, [])
