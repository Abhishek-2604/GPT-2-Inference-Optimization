# LLM Inference Optimization

A study of post-training optimization techniques applied to GPT-2 (124M). Each technique is implemented independently and evaluated on three axes: **model size**, **inference latency**, and **perplexity** on WikiText-2.

---

## Techniques

### 1. Post-Training Quantization (INT8)
Converts FP32 weights to INT8 without any retraining.

GPT-2 uses `Conv1D` layers internally (not standard `nn.Linear`), so standard `quantize_dynamic` targeting `nn.Linear` silently fails. We implement a custom `QuantizedConv1D` wrapper that correctly handles GPT-2's transposed weight layout `(in, out)` vs Linear's `(out, in)`:

```
scale = max(|W|) / 127
W_int8 = round(W / scale).clamp(-128, 127)
# At inference: W_dequant = W_int8.float() * scale
```

- 48 Conv1D layers quantized
- **51% size reduction** (474.7 MB → 231.8 MB)
- Minimal PPL degradation (56.1 → 65.0)
- Note: runtime dequantization on CPU adds overhead vs FP32 matmuls. Real speedup requires hardware INT8 kernels (TensorRT, bitsandbytes). This is the honest tradeoff of software-only PTQ.

### 2. Unstructured Magnitude Pruning
Zeroes out individual weights below a globally-computed magnitude threshold.

- Global threshold respects layer-wise weight distributions (per-layer thresholds over-prune sensitive layers)
- 30% of weights zeroed with only +2 PPL degradation (56.1 → 58.2)
- No latency improvement on CPU — zeroed weights still occupy dense tensors. Speedup requires sparse kernels (e.g. NVIDIA cuSPARSE). Demonstrates weight sensitivity without hardware dependency.

### 3. Knowledge Distillation
Trains a 4x smaller student (6 layers, 384 embd, ~30M params) to mimic GPT-2 teacher (12 layers, 768 embd, 124M params).

Combined loss balancing ground-truth and teacher supervision:
```
L = α · L_hard + (1 - α) · L_soft · T²
```
- `L_hard` — cross-entropy on ground-truth token labels
- `L_soft` — KL divergence between teacher/student softened distributions at temperature T
- `T²` rescales gradients suppressed by temperature division (Hinton et al., 2015)

---

## Results (measured on Colab T4, CPU inference)

| Method                   | Size (MB) | Params  | PPL  | Latency (ms)  | Speedup       |
|--------------------------|-----------|---------|------|---------------|---------------|
| Baseline (FP32)          | 474.7     | 124.4M  | 56.1 | 2885 ± 199    | —             |
| INT8 Quantization        | 231.8     | 39.5M*  | 65.0 | 8901 ± 188    | 0.32x†        |
| Unstructured Pruning 30% | 474.7     | 124.4M  | 58.2 | 3113 ± 318    | 0.93x†        |
| Distilled Student (~30M) | —         | ~30M    | —    | —             | ~2-3x (est.)  |

*Param count reflects quantized layer reporting; actual unique weights unchanged.  
†Slower on CPU due to software dequantization overhead. Hardware INT8 kernels (TensorRT/bitsandbytes) yield 1.5–2x speedup on GPU.

---

## File Layout

```
llm-inference-optimization/
├── model/
│   └── base.py              # GPT-2 loader, latency + perplexity measurement
├── quantization/
│   └── ptq.py               # Custom INT8 quantization for GPT-2's Conv1D layers
├── pruning/
│   └── prune.py             # Global magnitude unstructured pruning
├── distillation/
│   └── distill.py           # Teacher-student KL distillation training loop
├── benchmark.py             # Unified benchmark: all techniques, comparison table
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Quantization + pruning only (~15 min on Colab CPU)
python benchmark.py --skip_distill

# Full run including distillation (~45 min on Colab T4)
python benchmark.py --device cuda --distill_steps 1000

# Tune pruning aggressiveness
python benchmark.py --sparsity 0.5 --skip_distill
```

---

