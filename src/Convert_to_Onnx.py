#Convert to Onnx
import torch
import onnx
import onnxsim
import subprocess
import tensorflow as tf
from onnx_tf.backend import prepare

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "deeplabv3_mobilenet_fixed.onnx",
    opset_version=13,             # 11 is okay, but 13 works better with onnx-tf
    do_constant_folding=True,     # fold constants so weights don’t dangle
    input_names=["input"],        # explicit input name
    output_names=["output"],      # explicit output name
    dynamic_axes={                # allow variable batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    strict=True                   # ensures all params are exported
)

print("[1] Exported PyTorch → ONNX")

# --------------------------
# 2. Simplify ONNX
# --------------------------
model_onnx = onnx.load("deeplabv3_mobilenet_fixed.onnx")
model_simplified, check = onnxsim.simplify(model_onnx)

assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simplified, "deeplabv3_mobilenet_simplified.onnx")
print("[2] Simplified ONNX saved")

# --------------------------
# 3. Convert ONNX → TensorFlow (.pb)
# --------------------------
onnx_model = onnx.load("deeplabv3_mobilenet_simplified.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("deeplabv3_mobilenet.pb")
print("[3] Converted ONNX → TensorFlow (.pb)")