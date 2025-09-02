import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
step before 

onnx-tf convert -i deeplabv3_mobilenet.onnx -o deeplabv3_mobilenet.pb

"""

converter = tf.lite.TFLiteConverter.from_saved_model("D:\\Segmentation\\model\\deeplabv3_mobilenet.pb")

"""def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]
"""

#Quantize to float 16, mobile net has some layers which cano be quantized to int8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

"""converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8"""


tflite_model = converter.convert()

# Save model
with open("D:\\Segmentation\\model\\deeplabv3_mobilenet_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantized TFLite model saved: deeplabv3_mobilenet_quant.tflite")