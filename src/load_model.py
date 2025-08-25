import torch

from preprocess import *


model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()
#print(model)


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
output_predictions = preprocess_video(filename, model)


mask_image(filename, output_predictions)

