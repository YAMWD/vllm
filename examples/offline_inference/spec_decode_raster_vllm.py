# SPDX-License-Identifier: Apache-2.0
"""
Reproduce Figure 3 from "Fail Fast, Win Big" (Pan et al., 2025) using vLLM.

Approach:
  Phase 1 — Generate outputs using vLLM LLM with speculative_config (MTP).
            The scheduler is instrumented (via VLLM_SPEC_DECODE_TRACE_FILE)
            to log the actual per-token accept/reject decisions during
            speculative decoding. This captures the ground-truth spatial
            pattern of easy vs hard tokens.
  Phase 2 — Plot the raster.

Usage:
    python spec_decode_raster_vllm.py \
        --target Qwen/Qwen3.5-9B \
        --spec-len 4 --max-new-tokens 400 \
        --output figure3.png
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Default prompts (diverse difficulty)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    # 0 – coding (predictable boilerplate → easy regions)
    "Write a Python function that implements binary search on a sorted list. "
    "Include docstring, type hints, edge case handling, and unit tests.",

    # 1 – math step-by-step (formulaic but requires planning)
    "Solve step by step: A train leaves station A at 9 AM travelling at 80 km/h. "
    "Another train leaves station B (320 km away) at 10 AM travelling at 100 km/h "
    "toward station A. At what time do they meet? Show all work.",

    # 2 – summarization (lots of syntactic copying → easy)
    "Summarize the following text in detail:\n"
    "The Internet of Things (IoT) refers to the network of physical devices, "
    "vehicles, appliances and other objects embedded with sensors, software, and "
    "connectivity which enables these objects to connect and exchange data. The IoT "
    "allows objects to be sensed or controlled remotely across existing network "
    "infrastructure, creating opportunities for more direct integration of the "
    "physical world into computer-based systems.",

    # 3 – complex reasoning (hard regions expected)
    "There are 100 prisoners in solitary cells. A warden offers them a chance: "
    "he'll put cards numbered 1-100 in 100 boxes. Each prisoner may look in at "
    "most 50 boxes. If every prisoner finds their own number, all go free. "
    "What strategy maximises their probability of success and what is it?",

    # 4 – creative writing (mixed difficulty)
    "Write a short science fiction story (about 300 words) about the first "
    "contact between humanity and an alien civilisation that communicates "
    "entirely through mathematical proofs.",
]


# ---------------------------------------------------------------------------
# Phase 1: Generate with vLLM + speculative decoding (instrumented)
# ---------------------------------------------------------------------------

def generate_with_vllm(
    target_model: str,
    prompts: list[str],
    spec_len: int,
    max_new_tokens: int,
    dtype: str,
    trace_file: str,
) -> tuple[list[list[int]], dict]:
    """
    Generate outputs using vLLM with MTP speculative decoding.
    The scheduler writes per-token accept/reject traces to trace_file.

    Returns:
        output_token_ids : list of output token id lists
        metrics_summary  : dict of aggregate spec-decode metrics from vLLM
    """
    from vllm import LLM, SamplingParams

    # Tell the scheduler to write per-token acceptance traces.
    os.environ["VLLM_SPEC_DECODE_TRACE_FILE"] = trace_file

    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": spec_len,
    }

    llm = LLM(
        model=target_model,
        trust_remote_code=True,
        speculative_config=speculative_config,
        disable_log_stats=False,
        dtype=dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,          # greedy for reproducibility
        max_tokens=max_new_tokens,
    )

    print(f"\n{'='*60}")
    print(f"Phase 1: Generating with vLLM MTP speculative decoding")
    print(f"  target={target_model}")
    print(f"  spec_len={spec_len}  max_tokens={max_new_tokens}")
    print(f"  trace_file={trace_file}")
    print(f"{'='*60}\n")

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    output_token_ids = []
    request_ids = []
    for i, out in enumerate(outputs):
        o_ids = list(out.outputs[0].token_ids)
        output_token_ids.append(o_ids)
        request_ids.append(out.request_id)
        print(f"  Query {i}: prompt={len(out.prompt_token_ids)} tokens, "
              f"output={len(o_ids)} tokens  (req_id={out.request_id})")

    # Collect aggregate metrics
    metrics_summary = {}
    try:
        metrics = llm.get_metrics()
        for m in metrics:
            if "spec_decode" in m.name:
                metrics_summary[m.name] = (
                    m.values if hasattr(m, "values") else m.value
                )
    except Exception as e:
        print(f"  (could not read metrics: {e})")

    total_output = sum(len(o) for o in output_token_ids)
    print(f"\n  Total output tokens: {total_output}")
    print(f"  Wall time: {elapsed:.1f}s "
          f"({total_output / elapsed:.1f} tok/s)")
    if metrics_summary:
        drafts = metrics_summary.get("vllm:spec_decode_num_drafts", 0)
        accepted = metrics_summary.get(
            "vllm:spec_decode_num_accepted_tokens", 0
        )
        drafted = metrics_summary.get(
            "vllm:spec_decode_num_draft_tokens", 0
        )
        if drafted:
            print(f"  vLLM aggregate acceptance rate: "
                  f"{accepted}/{drafted} = {accepted/drafted:.2%}")
        if drafts:
            print(f"  Mean acceptance length: "
                  f"{1 + accepted / drafts:.2f}")

    # Clean up env var
    del os.environ["VLLM_SPEC_DECODE_TRACE_FILE"]

    # Free GPU memory
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return output_token_ids, request_ids, metrics_summary


# ---------------------------------------------------------------------------
# Read per-token acceptance trace from scheduler instrumentation
# ---------------------------------------------------------------------------

def read_acceptance_trace(
    trace_file: str,
    request_ids: list[str],
    output_token_ids: list[list[int]],
) -> list[list[bool]]:
    """
    Read the per-token acceptance trace file written by the scheduler.
    Each line is a JSON object: {"request_id": ..., "acceptance": [...]}.
    Returns acceptance masks in the same order as request_ids.
    """
    print(f"\n{'='*60}")
    print(f"Reading acceptance trace from {trace_file}")
    print(f"{'='*60}\n")

    # Parse JSONL trace file
    trace_data: dict[str, list[bool]] = {}
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                trace_data[entry["request_id"]] = entry["acceptance"]

    all_acceptance: list[list[bool]] = []
    for i, (req_id, o_ids) in enumerate(
        zip(request_ids, output_token_ids)
    ):
        # vLLM uses internal request IDs (e.g. "0-a679ea91") while the
        # external ID is just "0". Match by prefix.
        mask = None
        if req_id in trace_data:
            mask = trace_data[req_id]
        else:
            for trace_id, trace_mask in trace_data.items():
                if trace_id.startswith(req_id + "-") or \
                   trace_id == req_id:
                    mask = trace_mask
                    break
        if mask is not None:
            # Trim to actual output length (trace may include the first
            # prefill token which is part of the output).
            mask = mask[:len(o_ids)]
            # Pad if trace is shorter (shouldn't happen normally)
            while len(mask) < len(o_ids):
                mask.append(True)
        else:
            print(f"  WARNING: no trace for req_id={req_id}, "
                  f"marking all as accepted")
            mask = [True] * len(o_ids)

        n_acc = sum(mask)
        rate = n_acc / len(mask) if mask else 0
        print(f"  Query {i}: {len(mask)} tokens, "
              f"acceptance = {rate:.2%}  "
              f"(accepted={n_acc}, rejected={len(mask)-n_acc})")
        all_acceptance.append(mask)

    return all_acceptance


# ---------------------------------------------------------------------------
# Raster plot
# ---------------------------------------------------------------------------

ACCEPT_COLOR = (0 / 255, 100 / 255, 60 / 255)   # dark green (paper style)


def plot_raster(
    acceptance_list: list[list[bool]],
    query_ids: list[int],
    save_path: str = "figure3.png",
    title: str = "Varying difficulty of decoding within a sequence",
):
    max_len = max(len(a) for a in acceptance_list)
    n = len(acceptance_list)

    # Build RGB image (n rows x max_len cols)
    img = np.ones((n, max_len, 3), dtype=np.float32)  # white background
    for row, accepted in enumerate(acceptance_list):
        for col, acc in enumerate(accepted):
            if acc:
                img[row, col] = ACCEPT_COLOR

    fig, ax = plt.subplots(figsize=(10, 0.7 * n + 1.5))
    ax.imshow(img, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(n))
    ax.set_yticklabels(query_ids, fontsize=10)
    ax.set_xlabel("Output Token Index", fontsize=11)
    ax.set_ylabel("Query ID", fontsize=11)
    ax.set_title(title, fontsize=11, style="italic")

    green_patch = mpatches.Patch(
        color=ACCEPT_COLOR, label='accepted ("easier")'
    )
    white_patch = mpatches.Patch(
        facecolor="white", edgecolor="gray", label='rejected ("harder")'
    )
    ax.legend(
        handles=[green_patch, white_patch],
        loc="lower right", fontsize=9, framealpha=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[done] Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Figure 3 repro using vLLM MTP spec decode with "
                    "instrumented per-token acceptance tracing"
    )
    p.add_argument("--target", default="Qwen/Qwen3.5-9B")
    p.add_argument("--spec-len",       type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--num-prompts",    type=int, default=5)
    p.add_argument("--prompts-file",   default=None)
    p.add_argument("--output",         default="figure3_vllm.png")
    p.add_argument("--dtype",          default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--save-data",      default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- prompts ----
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS[: args.num_prompts]

    # ---- create temp trace file ----
    trace_fd, trace_file = tempfile.mkstemp(
        prefix="spec_decode_trace_", suffix=".jsonl"
    )
    os.close(trace_fd)

    # ---- Phase 1: vLLM generation with MTP spec decode ----
    output_token_ids, request_ids, metrics = generate_with_vllm(
        target_model=args.target,
        prompts=prompts,
        spec_len=args.spec_len,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        trace_file=trace_file,
    )

    # ---- Read ground-truth acceptance from scheduler trace ----
    all_acceptance = read_acceptance_trace(
        trace_file, request_ids, output_token_ids
    )

    query_ids = list(range(len(all_acceptance)))

    # ---- save raw data ----
    if args.save_data:
        data = {
            "target": args.target,
            "spec_len": args.spec_len,
            "method": "mtp (instrumented ground-truth)",
            "vllm_metrics": {
                k: (v if isinstance(v, (int, float)) else list(v))
                for k, v in metrics.items()
            },
            "queries": [
                {
                    "query_id": qid,
                    "output_len": len(output_token_ids[qid]),
                    "acceptance": [bool(a) for a in acc],
                    "acceptance_rate": (
                        sum(acc) / len(acc) if acc else 0
                    ),
                }
                for qid, acc in zip(query_ids, all_acceptance)
            ],
        }
        Path(args.save_data).write_text(json.dumps(data, indent=2))
        print(f"[done] Data saved to {args.save_data}")

    # ---- overall stats ----
    total = sum(len(a) for a in all_acceptance)
    agreed = sum(sum(a) for a in all_acceptance)
    print(f"\nOverall acceptance: {agreed}/{total} = {agreed/total:.2%}")

    # ---- Phase 2: plot ----
    plot_raster(
        all_acceptance,
        query_ids,
        save_path=args.output,
        title=(
            "Figure 3 repro — ground-truth MTP acceptance per token\n"
            f"(target: {args.target.split('/')[-1]}, "
            f"spec_len={args.spec_len})"
        ),
    )

    # Clean up trace file
    try:
        os.unlink(trace_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
