"""
Post-Training Quantization for GPT-2.
GPT-2 uses transformers.pytorch_utils.Conv1D internally, not nn.Linear.
We quantize by manually casting weights to int8 and dequantizing at runtime.
"""

import copy
import io
import torch
import torch.nn as nn
try:
    from transformers.pytorch_utils import Conv1D
    HAS_CONV1D = True
except ImportError:
    HAS_CONV1D = False


class QuantizedConv1D(nn.Module):
    """
    INT8 quantized wrapper for GPT-2's Conv1D.
    Conv1D weight shape is (in_features, out_features) — transposed vs nn.Linear.
    """
    def __init__(self, conv1d_layer):
        super().__init__()
        weight = conv1d_layer.weight.data.float()  # (in, out)
        self.scale = weight.abs().max() / 127.0
        w_int8 = (weight / self.scale).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer("w_int8", w_int8)
        self.bias = conv1d_layer.bias
        self.nf = conv1d_layer.nf

    def forward(self, x):
        w = self.w_int8.float() * self.scale
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        return out.view(size_out)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Quantize all Conv1D layers in GPT-2 to INT8."""
    model_q = copy.deepcopy(model).cpu().eval()

    def replace_conv1d(module):
        for name, child in module.named_children():
            if HAS_CONV1D and isinstance(child, Conv1D):
                setattr(module, name, QuantizedConv1D(child))
            else:
                replace_conv1d(child)

    replace_conv1d(model_q)

    # Count replaced layers
    n_quantized = sum(1 for m in model_q.modules() if isinstance(m, QuantizedConv1D))
    print(f"  Quantized {n_quantized} Conv1D layers to INT8")
    return model_q


def model_size_mb_quantized(model: nn.Module) -> float:
    """Measure real serialized size."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
