import torch
import tensorflow as tf

from preprocess import *


interpreter = tf.lite.Interpreter(model_path="D:\\Segmentation\\src\\deeplabv3_mobilenet_quant.tflite")
#interpreter.eval()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details:", input_details)
print("output_details:", output_details)
breakpoint()



#import glob
#from PIL import Image
#filename = "C:\\Users\\Vivupadi\\Desktop\\Portfolio\\img\\section_2.jpg"
filename = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250802_072807827.mp4"



#output_predictions = preprocess_image(filename, model)
#output_predictions = preprocess_video(filename, model)
output_predictions = preprocess_video(filename, interpreter)


mask_image(filename, output_predictions)

