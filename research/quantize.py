"""Per-channel dynamic int8 quantization of model.onnx (MatMul/Gemm weights only).
The earlier per-tensor variant moved logits by up to 5.0; per-channel keeps a scale per
output channel. Accuracy is decided by research/bench.py, not here."""
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType
src, dst = sys.argv[1], sys.argv[2]
rr = len(sys.argv) > 3 and sys.argv[3] == "--reduce-range"
quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, reduce_range=rr,
                 op_types_to_quantize=["MatMul", "Gemm"])
print("wrote", dst)
