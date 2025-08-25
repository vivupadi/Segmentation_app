import onnx
from onnx_tf.backend import prepare
from onnxsim import simplify
from onnx import utils

# Load ONNX model
onnx_model = onnx.load("mobilenet_renamed.onnx")

#breakpoint()
# Check validity
#sorted_model = utils.polish_model(onnx_model)
#onnx.save(sorted_model, "sorted.onnx")
#print("Saved fixed model as sorted.onnx")
#onnx.checker.check_model(onnx_model )
#breakpoint()
# Simplify + re-sort nodes
model_simplified, check = simplify(onnx_model )

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simplified, "simplified.onnx")
print("Simplified model saved to simplified.onnx")


