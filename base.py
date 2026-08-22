"""
Base model: GPT-2 (124M) loaded from HuggingFace.
Provides baseline latency, size, and perplexity measurements.
"""

import time
import math
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset


def load_gpt2(device: str = "cpu") -> tuple:
    """Load pretrained GPT-2 small and tokenizer."""
    print("Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval().to(device)
    print(f"  Parameters:      {count_params(model):,}")
    print(f"  Size on disk:    {model_size_mb(model):.1f} MB")
    return model, tokenizer


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB from parameter dtypes."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def measure_latency(
    model: nn.Module,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    n_runs: int = 20,
    device: str = "cpu",
) -> dict:
    """
    Measure mean + std generation latency over n_runs.
    Returns dict with mean_ms, std_ms, tokens_per_sec.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(latencies) / len(latencies)
    variance = sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)
    std_ms = variance ** 0.5
    tokens_per_sec = (max_new_tokens / mean_ms) * 1000

    return {"mean_ms": mean_ms, "std_ms": std_ms, "tokens_per_sec": tokens_per_sec}


def measure_perplexity(
    model: nn.Module,
    tokenizer,
    n_batches: int = 20,
    seq_len: int = 128,
    device: str = "cpu",
) -> float:
    """
    Estimate perplexity on WikiText-2 validation set.
    Uses non-overlapping windows for speed.
    """
    print("  Computing perplexity on WikiText-2 val...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    total_loss = 0.0
    total_tokens = 0
    batches_done = 0

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - seq_len, seq_len):
            if batches_done >= n_batches:
                break
            chunk = input_ids[:, i : i + seq_len]
            if chunk.size(1) < seq_len:
                break
            outputs = model(chunk, labels=chunk)
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len
            batches_done += 1

    return math.exp(total_loss / total_tokens)
