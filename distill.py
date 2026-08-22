"""
Knowledge Distillation: GPT-2 (124M) Teacher → GPT-2 Student (custom smaller config).

Two loss components:
  1. Hard loss  — standard cross-entropy on ground-truth labels
  2. Soft loss  — KL divergence between teacher and student output distributions
                  (teacher logits divided by temperature T to soften probabilities)

Combined loss:
  L = α * L_hard + (1 - α) * L_soft * T²

The T² factor restores the gradient magnitude that gets suppressed by dividing
logits by T (Hinton et al., 2015).

Architecture:
  Teacher: GPT-2 small  — 12 layers, 12 heads, 768 embd   (124M params)
  Student: GPT-2 mini   —  6 layers,  6 heads, 384 embd   (~30M params)
"""

import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from datasets import load_dataset


# ── Student Model ─────────────────────────────────────────────────────────────

def build_student(device: str = "cpu") -> nn.Module:
    """
    Build a smaller GPT-2 student from scratch (random init).
    Half the layers and heads, half the embedding dimension vs GPT-2 small.
    """
    student_cfg = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=384,
        n_layer=6,
        n_head=6,
        n_inner=384 * 4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
    )
    student = GPT2LMHeadModel(student_cfg)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    return student.to(device)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WikiTextChunks(Dataset):
    """Fixed-length token chunks from WikiText-103 for distillation training."""

    def __init__(self, tokenizer, split: str = "train", seq_len: int = 128, max_chunks: int = 5000):
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)
        text = " ".join(ex["text"] for ex in dataset if ex["text"].strip())
        tokens = tokenizer.encode(text)
        self.chunks = [
            tokens[i : i + seq_len]
            for i in range(0, len(tokens) - seq_len, seq_len)
        ][:max_chunks]
        print(f"  Distillation dataset: {len(self.chunks):,} chunks ({split})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


# ── Distillation Loss ─────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple:
    """
    Compute combined distillation loss.

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V)  — detached, no grad
        labels:         (B, T)     — ground-truth token ids
        temperature:    T for softening distributions
        alpha:          weight of hard loss (1-alpha = soft loss weight)

    Returns:
        (total_loss, hard_loss, soft_loss)
    """
    B, T, V = student_logits.shape

    # Hard loss: cross-entropy on ground-truth labels
    hard_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-1,
    )

    # Soft loss: KL divergence with temperature-softened distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student) — we want student to match teacher
    soft_loss = F.kl_div(
        student_soft.view(-1, V),
        teacher_soft.view(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)  # rescale gradients

    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss, hard_loss, soft_loss


# ── Training Loop ─────────────────────────────────────────────────────────────

def distill(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    device: str = "cpu",
    n_steps: int = 1000,
    batch_size: int = 8,
    lr: float = 3e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    seq_len: int = 128,
    log_interval: int = 100,
) -> nn.Module:
    """
    Run knowledge distillation: teacher supervises student training.

    The teacher is frozen. The student learns from both:
      - Ground-truth next-token labels (hard targets)
      - Teacher's softened output distributions (soft targets)
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    dataset = WikiTextChunks(tokenizer, split="train", seq_len=seq_len + 1, max_chunks=n_steps * batch_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    print(f"\nDistilling: {n_steps} steps | T={temperature} | α={alpha}")
    t0 = time.time()

    for step in range(n_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x = batch[:, :-1].to(device)   # input
        y = batch[:, 1:].to(device)    # labels (shifted by 1)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(x)
            teacher_logits = teacher_out.logits  # (B, T, V)

        # Student forward
        student_out = student(x)
        student_logits = student_out.logits  # (B, T, V)

        loss, hard, soft = distillation_loss(
            student_logits, teacher_logits, y, temperature, alpha
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss {loss.item():.4f} | "
                  f"hard {hard.item():.4f} | soft {soft.item():.4f} | "
                  f"{elapsed:.0f}s elapsed")
            t0 = time.time()

    student.eval()
    return student
