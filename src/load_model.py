import torch
import tensorflow as tf

from preprocess import *


#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
#model= 'D:\\Segmentation\\src\\deeplabv3_mobilenet_quant.tflite'
interpreter = tf.lite.Interpreter(model_path="D:\\Segmentation\\src\\deeplabv3_mobilenet_quant.tflite")
#interpreter.eval()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details:", input_details)
print("output_details:", output_details)
#print(model)
breakpoint()


####Test file###
# Download an example image from the pytorch website
#import urllib
#url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
##try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)
################

#import glob
#from PIL import Image
#filename = "C:\\Users\\Vivupadi\\Desktop\\Portfolio\\img\\section_2.jpg"
filename = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250802_072807827.mp4"



#output_predictions = preprocess_image(filename, model)
#output_predictions = preprocess_video(filename, model)
output_predictions = preprocess_video(filename, interpreter)


mask_image(filename, output_predictions)

