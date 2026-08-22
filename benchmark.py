"""
Unified Benchmark: GPT-2 Optimization Techniques
Compares baseline vs quantization vs pruning vs distillation
on three axes: latency, model size, and perplexity.

Usage:
    python benchmark.py                          # all techniques, CPU
    python benchmark.py --device cuda            # run on GPU
    python benchmark.py --skip_distill           # skip distillation (slow)
    python benchmark.py --sparsity 0.5           # 50% pruning
"""

import os
import sys
import math
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model.base import (
    load_gpt2,
    measure_latency,
    measure_perplexity,
    model_size_mb,
    count_params,
)
from quantization.ptq import apply_dynamic_quantization, model_size_mb_quantized
from pruning.prune import apply_unstructured_pruning, prune_attention_heads, actual_sparsity
from distillation.distill import build_student, distill


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="LLM Inference Optimization Benchmark")
    p.add_argument("--device",        type=str,   default="cpu",  help="cpu or cuda")
    p.add_argument("--sparsity",      type=float, default=0.3,    help="Unstructured pruning sparsity")
    p.add_argument("--prune_ratio",   type=float, default=0.25,   help="Fraction of attention heads to prune")
    p.add_argument("--distill_steps", type=int,   default=500,    help="Distillation training steps")
    p.add_argument("--max_new_tokens",type=int,   default=50,     help="Tokens to generate per latency run")
    p.add_argument("--n_runs",        type=int,   default=20,     help="Latency measurement repetitions")
    p.add_argument("--n_ppl_batches", type=int,   default=20,     help="Batches for perplexity estimation")
    p.add_argument("--skip_distill",  action="store_true",        help="Skip distillation (saves time)")
    p.add_argument("--prompt",        type=str,   default="The history of artificial intelligence")
    return p.parse_args()


# ── Result Formatting ─────────────────────────────────────────────────────────

def fmt_speedup(baseline_ms: float, model_ms: float) -> str:
    s = baseline_ms / model_ms
    return f"{s:.2f}x {'faster' if s >= 1 else 'slower'}"


def print_results(results: list):
    """Print a formatted comparison table."""
    col_w = [28, 12, 12, 12, 14, 12]
    headers = ["Method", "Size (MB)", "Params", "PPL", "Latency (ms)", "Speedup"]

    sep = "─" * sum(col_w)
    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_w))

    print(f"\n{'=' * sum(col_w)}")
    print("  BENCHMARK RESULTS")
    print(f"{'=' * sum(col_w)}")
    print(header_row)
    print(sep)

    baseline_ms = None
    for r in results:
        if baseline_ms is None:
            baseline_ms = r["latency_ms"]
        speedup = fmt_speedup(baseline_ms, r["latency_ms"])
        ppl_str = f"{r['perplexity']:.1f}" if r["perplexity"] else "—"
        row = [
            r["name"],
            f"{r['size_mb']:.1f}",
            f"{r['params']/1e6:.1f}M",
            ppl_str,
            f"{r['latency_ms']:.1f} ± {r['latency_std']:.1f}",
            speedup if r["name"] != "Baseline (FP32)" else "—",
        ]
        print("".join(str(c).ljust(w) for c, w in zip(row, col_w)))

    print(f"{'=' * sum(col_w)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = args.device
    results = []

    print(f"\n{'='*60}")
    print(f"  LLM Inference Optimization Benchmark")
    print(f"  Device: {device.upper()}")
    print(f"{'='*60}\n")

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    print("── [1/4] Baseline (FP32) ──────────────────────────────────")
    model, tokenizer = load_gpt2(device)

    lat = measure_latency(model, tokenizer, args.prompt, args.max_new_tokens, args.n_runs, device)
    ppl = measure_perplexity(model, tokenizer, args.n_ppl_batches, device=device)

    results.append({
        "name":        "Baseline (FP32)",
        "size_mb":     model_size_mb(model),
        "params":      count_params(model),
        "perplexity":  ppl,
        "latency_ms":  lat["mean_ms"],
        "latency_std": lat["std_ms"],
        "tok_per_sec": lat["tokens_per_sec"],
    })
    print(f"  Baseline PPL: {ppl:.2f} | Latency: {lat['mean_ms']:.1f}ms | "
          f"{lat['tokens_per_sec']:.0f} tok/s\n")

    # ── 2. Dynamic INT8 Quantization ─────────────────────────────────────────
    print("── [2/4] Dynamic INT8 Quantization ────────────────────────")
    # Reload fresh FP32 model before quantizing to avoid any state contamination
    model_q_base, _ = load_gpt2("cpu")
    model_q = apply_dynamic_quantization(model_q_base)

    lat_q = measure_latency(model_q, tokenizer, args.prompt, args.max_new_tokens, args.n_runs, "cpu")
    ppl_q = measure_perplexity(model_q, tokenizer, args.n_ppl_batches, device="cpu")

    results.append({
        "name":        "INT8 Quantization",
        "size_mb":     model_size_mb_quantized(model_q),
        "params":      count_params(model_q),
        "perplexity":  ppl_q,
        "latency_ms":  lat_q["mean_ms"],
        "latency_std": lat_q["std_ms"],
        "tok_per_sec": lat_q["tokens_per_sec"],
    })
    print(f"  Quantized PPL: {ppl_q:.2f} | Latency: {lat_q['mean_ms']:.1f}ms | "
          f"{lat_q['tokens_per_sec']:.0f} tok/s\n")

    # ── 3a. Unstructured Pruning ──────────────────────────────────────────────
    print(f"── [3/4] Pruning ({args.sparsity:.0%} unstructured + {args.prune_ratio:.0%} heads) ──")
    model_pruned = apply_unstructured_pruning(model, sparsity=args.sparsity)

    lat_p = measure_latency(model_pruned, tokenizer, args.prompt, args.max_new_tokens, args.n_runs, "cpu")
    ppl_p = measure_perplexity(model_pruned, tokenizer, args.n_ppl_batches, device="cpu")

    results.append({
        "name":        f"Unstructured Pruning ({args.sparsity:.0%})",
        "size_mb":     model_size_mb(model_pruned),
        "params":      count_params(model_pruned),
        "perplexity":  ppl_p,
        "latency_ms":  lat_p["mean_ms"],
        "latency_std": lat_p["std_ms"],
        "tok_per_sec": lat_p["tokens_per_sec"],
    })
    print(f"  Pruned PPL: {ppl_p:.2f} | Latency: {lat_p['mean_ms']:.1f}ms | "
          f"Actual sparsity: {actual_sparsity(model_pruned):.1%}\n")


    # ── 4. Knowledge Distillation ─────────────────────────────────────────────
    if not args.skip_distill:
        print("── [4/4] Knowledge Distillation ────────────────────────────")
        teacher = model
        student = build_student(device)

        student = distill(
            teacher, student, tokenizer,
            device=device,
            n_steps=args.distill_steps,
        )

        lat_d = measure_latency(student, tokenizer, args.prompt, args.max_new_tokens, args.n_runs, device)
        ppl_d = measure_perplexity(student, tokenizer, args.n_ppl_batches, device=device)

        results.append({
            "name":        f"Distilled Student (~30M)",
            "size_mb":     model_size_mb(student),
            "params":      count_params(student),
            "perplexity":  ppl_d,
            "latency_ms":  lat_d["mean_ms"],
            "latency_std": lat_d["std_ms"],
            "tok_per_sec": lat_d["tokens_per_sec"],
        })
        print(f"  Student PPL: {ppl_d:.2f} | Latency: {lat_d['mean_ms']:.1f}ms | "
              f"{lat_d['tokens_per_sec']:.0f} tok/s\n")
    else:
        print("── [4/4] Distillation skipped (--skip_distill)\n")

    # ── Results Table ──────────────────────────────────────────────────────────
    print_results(results)

    # ── Save results to file ───────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.txt", "w") as f:
        f.write(f"Device: {device}\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"  Size:      {r['size_mb']:.1f} MB\n")
            f.write(f"  Params:    {r['params']/1e6:.1f}M\n")
            f.write(f"  PPL:       {r['perplexity']:.2f}\n")
            f.write(f"  Latency:   {r['latency_ms']:.1f} ± {r['latency_std']:.1f} ms\n")
            f.write(f"  Tok/sec:   {r['tok_per_sec']:.0f}\n\n")
    print("Results saved to results/benchmark.txt")


if __name__ == "__main__":
    main()
