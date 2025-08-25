import tensorflow as tf
from tensorflow import keras

"""
step before 

onnx-tf convert -i deeplabv3_mobilenet.onnx -o deeplabv3_mobilenet.pb

"""

# Load SavedModel
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file="deeplabv3_mobilenet.pb",
    input_arrays=["input"],       # must match the names in your graph
    output_arrays=["output"],     # must match the names in your graph
    input_shapes={"input": [1, 3, 224, 224]}
)

# Dynamic Range Quantization (smaller + faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()

# Save model
with open("deeplabv3_mobilenet_quant.tflite", "wb") as f:
    f.write(tflite_model)