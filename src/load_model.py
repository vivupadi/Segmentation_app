import torch

from preprocess import *


model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()
#print(model)


####Test file###
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
################


output_predictions = preprocess_image(filename, model)


mask_image(filename, output_predictions)

dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "deeplabv3_mobilenet.onnx",
    opset_version=11
)