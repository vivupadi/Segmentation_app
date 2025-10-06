# Image Segmentation on Videos
A realtime frame-by-frame segmentation on user-given video input.

## Model Used
Deeplabv3-MobileNetV3-Large is constructed by a Deeplabv3 model using the MobileNetV3 large backbone. The pre-trained model has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.

The accuracies of the pre-trained model evaluated on COCO val2017 dataset are listed below.
<pre>
Model structure	              Mean IOU	Global Pixelwise Accuracy
deeplabv3_mobilenet_v3_large	  60.3	          91.2
</pre>

## Tech Stack

## Model architecture

## Inference on CPU
![til](https://github.com/vivupadi/Segmentation_app/blob/main/data/Normal_trimmed.gif)

## Quantization


## Quantized model(float 16) inference on local system
![til](https://github.com/vivupadi/Segmentation_app/blob/main/data/quantized_trimmed.gif)

## Quantized model(float 16) inference on Raspberry Pi 4

# Future Plans

## Reduce model to segment only Humans

## Inference on Raspberry Pi 4 + Camera module


## Structure of Pipeline


<div align="center">
⭐ Star this repo if you find it helpful!
  
Made with ❤️ by Vivek Padayattil
</div>
