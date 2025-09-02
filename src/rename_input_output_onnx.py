import onnx
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model = onnx.load("D:\\Segmentation\\model\\deeplabv3_mobilenet.onnx")

for inp in onnx_model.graph.input:
    print(inp.name)

# Rename all graph inputs
for inp in onnx_model.graph.input:
    if "." in inp.name:
        inp.name = inp.name.replace(".", "_")  # e.g. input.1 → input_1

# Rename node references
for node in onnx_model.graph.node:
    node.input[:] = [i.replace(".", "_") for i in node.input]
    node.output[:] = [o.replace(".", "_") for o in node.output]

onnx.save(onnx_model, "D:\\Segmentation\\model\\mobilenet_renamed.onnx")

