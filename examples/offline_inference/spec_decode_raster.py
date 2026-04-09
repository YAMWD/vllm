# SPDX-License-Identifier: Apache-2.0
"""
Reproduce Figure 3 from "Fail Fast, Win Big" (Pan et al., 2025).

Runs speculative decoding with per-token acceptance tracking and plots
a raster where green = drafter accepted, white = drafter rejected.

Usage:
    python spec_decode_raster.py \
        --target Qwen/Qwen2.5-7B-Instruct \
        --draft  Qwen/Qwen2.5-1.5B-Instruct \
        --spec-len 4 --max-new-tokens 512 \
        --output figure3.png
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Prompts (diverse: coding / math / reasoning / summarization / QA)
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
# Speculative decoding with acceptance tracking
# ---------------------------------------------------------------------------

def sample_from_probs(probs: torch.Tensor) -> int:
    """Multinomial sample from a probability vector."""
    return int(torch.multinomial(probs, num_samples=1).item())


def draft_with_probs(
    model, input_ids: torch.Tensor, k: int
) -> tuple[list[int], list[torch.Tensor]]:
    """
    Auto-regressively generate k tokens from draft model.
    Returns (token_ids, prob_distributions) where prob_distributions[i]
    is the full softmax distribution q(·) at step i.
    """
    tokens: list[int] = []
    probs:  list[torch.Tensor] = []
    ids  = input_ids.clone()
    past = None
    for _ in range(k):
        out  = model(ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logit = out.logits[0, -1]                  # (vocab,)
        q     = torch.softmax(logit, dim=-1)       # draft distribution q
        tok   = sample_from_probs(q)               # sample (not greedy)
        tokens.append(tok)
        probs.append(q)
        ids = torch.tensor([[tok]], device=ids.device)
    return tokens, probs


def speculative_decode_tracked(
    target_model,
    draft_model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 512,
    spec_len: int = 4,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[list[int], list[bool]]:
    """
    Speculative decoding with full rejection sampling (Leviathan et al., 2023).

    At each position i the draft token x_i is accepted with probability:
        min(1,  p_target(x_i) / p_draft(x_i))

    On rejection, a corrected token is sampled from the residual distribution:
        norm( max(0,  p_target - p_draft) )

    This guarantees the output distribution is identical to sampling from
    the target model directly (lossless).

    Returns:
        generated_tokens : list of output token ids
        acceptance_mask  : True  = draft token accepted at this position
                           False = draft rejected; target's residual used
    """
    generated: list[int] = list(prompt_ids)
    acceptance: list[bool] = []
    eos = tokenizer.eos_token_id

    with torch.no_grad():
        while len(acceptance) < max_new_tokens:
            remaining = max_new_tokens - len(acceptance)
            k = min(spec_len, remaining)

            # ── 1. Draft: sample k tokens, keep their distributions ──────────
            draft_input  = torch.tensor([generated], device=device)
            draft_tokens, draft_probs = draft_with_probs(draft_model, draft_input, k)

            # ── 2. Verify: ONE target forward pass over context + draft ───────
            verify_ids    = torch.tensor([generated + draft_tokens], device=device)
            target_logits = target_model(verify_ids).logits[0]
            # Slice out the k+1 positions we care about:
            #   position len(generated)-1  → target's distribution given context only
            #   position len(generated)+i  → target's distribution given context+draft[:i]
            verify_logits = target_logits[len(generated) - 1:]   # (k+1, vocab)
            target_probs  = torch.softmax(verify_logits, dim=-1)  # p_target at each pos

            # ── 3. Accept / reject each draft token (rejection sampling) ──────
            n_accepted = 0
            for i, (dt, q_i) in enumerate(zip(draft_tokens, draft_probs)):
                p_i = target_probs[i]          # target distribution at position i
                # acceptance probability: min(1, p_target(x_i) / q_draft(x_i))
                accept_prob = min(1.0, (p_i[dt] / (q_i[dt] + 1e-9)).item())

                if torch.rand(1).item() < accept_prob:
                    # ── accepted ──
                    generated.append(dt)
                    acceptance.append(True)
                    n_accepted += 1
                    if dt == eos:
                        return generated[len(prompt_ids):], acceptance
                else:
                    # ── rejected: sample from residual distribution ──
                    # residual = norm( max(0, p_target - q_draft) )
                    residual = torch.clamp(p_i - q_i, min=0.0)
                    residual_sum = residual.sum()
                    if residual_sum < 1e-9:
                        # degenerate: fall back to target distribution
                        residual = p_i.clone()
                        residual_sum = residual.sum()
                    corrected = sample_from_probs(residual / residual_sum)
                    generated.append(corrected)
                    acceptance.append(False)
                    if corrected == eos:
                        return generated[len(prompt_ids):], acceptance
                    break  # remaining draft tokens are discarded

            # ── 4. Bonus token when all k drafts accepted ─────────────────────
            if n_accepted == k:
                # sample from target's distribution at position k
                bonus = sample_from_probs(target_probs[k])
                generated.append(bonus)
                acceptance.append(True)
                if bonus == eos:
                    return generated[len(prompt_ids):], acceptance

    return generated[len(prompt_ids):], acceptance


# ---------------------------------------------------------------------------
# Raster plot
# ---------------------------------------------------------------------------

ACCEPT_COLOR = (0 / 255, 100 / 255, 60 / 255)   # dark green  (paper style)
REJECT_COLOR = (1.0, 1.0, 1.0)                   # white


def plot_raster(
    acceptance_list: list[list[bool]],
    query_ids: list[int],
    save_path: str = "figure3.png",
    title: str = "Varying difficulty of decoding within a sequence",
):
    max_len = max(len(a) for a in acceptance_list)
    n = len(acceptance_list)

    # Build RGB image (n rows × max_len cols)
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

    green_patch = mpatches.Patch(color=ACCEPT_COLOR, label='accepted ("easier")')
    white_patch = mpatches.Patch(
        facecolor="white", edgecolor="gray", label='rejected ("harder")'
    )
    ax.legend(
        handles=[green_patch, white_patch],
        loc="lower right", fontsize=9, framealpha=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Spec-decode raster plot (Figure 3 repro)")
    p.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model ID or local path for target model")
    p.add_argument("--draft",  default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID or local path for draft model")
    p.add_argument("--spec-len",      type=int,   default=4,
                   help="Number of tokens to speculate per round (default: 4)")
    p.add_argument("--max-new-tokens", type=int,  default=400,
                   help="Max output tokens per prompt (default: 400)")
    p.add_argument("--num-prompts",   type=int,   default=5,
                   help="How many prompts to run (1-5, uses built-in set)")
    p.add_argument("--prompts-file",  default=None,
                   help="Optional JSON file with a list of prompt strings")
    p.add_argument("--output", default="figure3_repro.png",
                   help="Output PNG path")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--save-data", default=None,
                   help="Optional path to save acceptance data as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = {"float16": torch.float16,
              "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]

    # ---- prompts ----
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS[: args.num_prompts]

    print(f"Target : {args.target}")
    print(f"Draft  : {args.draft}")
    print(f"Prompts: {len(prompts)}  |  spec_len={args.spec_len}"
          f"  |  max_new_tokens={args.max_new_tokens}")

    # ---- load tokenizer ----
    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- load models ----
    print("Loading target model …")
    t0 = time.time()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()
    print(f"  target loaded in {time.time()-t0:.1f}s")

    print("Loading draft model …")
    t0 = time.time()
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    draft_model.eval()
    print(f"  draft  loaded in {time.time()-t0:.1f}s\n")

    # ---- run spec decode ----
    all_acceptance: list[list[bool]] = []
    query_ids: list[int] = []

    for idx, prompt in enumerate(prompts):
        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        print(f"Query {idx} | prompt tokens={len(prompt_ids)} …", end=" ", flush=True)
        t0 = time.time()

        _, acceptance = speculative_decode_tracked(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            spec_len=args.spec_len,
            device=device,
        )

        elapsed = time.time() - t0
        n_accepted = sum(acceptance)
        acc_rate = n_accepted / len(acceptance) if acceptance else 0
        print(
            f"generated={len(acceptance)} tokens | "
            f"accept_rate={acc_rate:.2%} | {elapsed:.1f}s"
        )

        all_acceptance.append(acceptance)
        query_ids.append(idx)

    # ---- save raw data ----
    if args.save_data:
        data = {
            "target": args.target,
            "draft": args.draft,
            "spec_len": args.spec_len,
            "queries": [
                {"query_id": qid, "acceptance": [bool(a) for a in acc]}
                for qid, acc in zip(query_ids, all_acceptance)
            ],
        }
        Path(args.save_data).write_text(json.dumps(data, indent=2))
        print(f"[✓] Raw acceptance data saved to {args.save_data}")

    # ---- overall stats ----
    total_tokens   = sum(len(a) for a in all_acceptance)
    total_accepted = sum(sum(a) for a in all_acceptance)
    print(f"\nOverall acceptance rate: "
          f"{total_accepted}/{total_tokens} = {total_accepted/total_tokens:.2%}")
    print(f"Mean acceptance length (MAL): "
          f"{1 + total_accepted / max(1, total_tokens - total_accepted):.2f}")

    # ---- plot ----
    plot_raster(
        all_acceptance,
        query_ids,
        save_path=args.output,
        title=(
            "Figure 3 repro — varying difficulty of decoding within a sequence\n"
            f"(target: {args.target.split('/')[-1]}, "
            f"draft: {args.draft.split('/')[-1]}, spec_len={args.spec_len})"
        ),
    )


if __name__ == "__main__":
    main()
