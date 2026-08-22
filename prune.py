"""
Weight Pruning for LLM Inference Optimization.
"""

import copy
import torch
import torch.nn as nn


def get_global_magnitude_threshold(model, sparsity):
    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.abs().flatten())
    all_weights = torch.cat(all_weights)
    k = int(sparsity * all_weights.numel())
    return all_weights.kthvalue(max(k, 1)).values.item()


def apply_unstructured_pruning(model, sparsity=0.3):
    model_p = copy.deepcopy(model).cpu()
    threshold = get_global_magnitude_threshold(model_p, sparsity)
    masks = {}
    for name, module in model_p.named_modules():
        if isinstance(module, nn.Linear):
            mask = module.weight.data.abs() >= threshold
            module.weight.data *= mask.float()
            masks[name] = mask
    model_p.pruning_masks = masks
    actual = actual_sparsity(model_p)
    print(f"  Unstructured pruning — target: {sparsity:.0%} | actual: {actual:.2%}")
    return model_p


def actual_sparsity(model):
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def score_attention_heads(model):
    scores = {}
    for i, block in enumerate(model.transformer.h):
        W = block.attn.c_attn.weight  # (3*n_embd, n_embd)
        n_embd = model.config.n_embd
        n_head = model.config.n_head
        head_dim = n_embd // n_head
        W_q = W[:n_embd, :]
        W_q_heads = W_q.reshape(n_head, head_dim, -1)
        head_scores = W_q_heads.abs().sum(dim=(1, 2))
        scores[i] = head_scores.detach().cpu()
    return scores


def _zero_pruned_heads(model, heads_to_prune):
    """Zero out Q/K/V weights for pruned heads (structured sparsity without changing architecture)."""
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    for layer_idx, head_indices in heads_to_prune.items():
        block = model.transformer.h[layer_idx]
        W = block.attn.c_attn.weight.data  # (3*n_embd, n_embd)
        for head_idx in head_indices:
            for offset in [0, n_embd, 2 * n_embd]:  # Q, K, V slices
                start = offset + head_idx * head_dim
                end   = start + head_dim
                W[start:end, :] = 0.0


def prune_attention_heads(model, prune_ratio=0.25):
    model_p = copy.deepcopy(model).cpu()
    n_head  = model_p.config.n_head
    n_prune = max(1, int(prune_ratio * n_head))

    scores = score_attention_heads(model_p)
    heads_to_prune = {}
    for layer_idx, head_scores in scores.items():
        _, sorted_idx = head_scores.sort()
        heads_to_prune[layer_idx] = sorted_idx[:n_prune].tolist()

    total_pruned = sum(len(v) for v in heads_to_prune.values())
    total_heads  = n_head * len(scores)
    print(f"  Structured pruning — {total_pruned}/{total_heads} heads pruned "
          f"({total_pruned/total_heads:.0%} of attention heads)")

    _zero_pruned_heads(model_p, heads_to_prune)
    return model_p, heads_to_prune


def sparse_storage_size_mb(model):
    total_bytes = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            nnz = (w != 0).sum().item()
            total_bytes += nnz * (4 + 4)
    return total_bytes / (1024 ** 2)
