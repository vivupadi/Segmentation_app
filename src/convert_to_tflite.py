import onnx
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model = onnx.load("deeplabv3_mobilenet.onnx")

# Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)

# Export as SavedModel (needed for TFLite conversion)
saved_model_dir = "./saved_model"
tf_rep.export_graph(saved_model_dir)