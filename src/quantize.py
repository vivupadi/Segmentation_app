import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
step before 

onnx-tf convert -i deeplabv3_mobilenet.onnx -o deeplabv3_mobilenet.pb

"""

"""# Load SavedModel
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file="deeplabv3_mobilenet.pb",
    input_arrays=["input"],       # must match the names in your graph
    output_arrays=["output"],     # must match the names in your graph
    input_shapes={"input": [1, 3, 224, 224]}
)"""


converter = tf.lite.TFLiteConverter.from_saved_model("deeplabv3_mobilenet.pb")

"""def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]
"""

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

"""converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8"""


tflite_model = converter.convert()

# Save model
with open("deeplabv3_mobilenet_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantized TFLite model saved: deeplabv3_mobilenet_quant.tflite")