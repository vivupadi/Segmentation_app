import torch

from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(filename, model):
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    return output.argmax(0) 


def preprocess_video(filename, model):
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, fFrame = video.read()
        if not ret:
            break
        
        input_image = cv2.cvtColor(fFrame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (224, 224))
        #input_image = Image.open(filename)

        #get frames
        #input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
            mask_video(fFrame, output.argmax(0))
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
    #return mask_video(fFrame, output.argmax(0))


def mask_image(filename, output_predictions):
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.imshow(r)
    plt.show()


#Masking video
def mask_video(fFrame, output_predictions):
    #video = cv2.VideoCapture(filename)
    #ret, fFrame = video.read()

    input_image = cv2.cvtColor(fFrame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224))
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    breakpoint()
    h,w, _ = fFrame.shape
    # plot the semantic segmentation predictions of 21 classes in each color
    mask = output_predictions.byte().cpu().numpy()
    r = Image.fromarray(mask).resize((w,h))

    # Apply color map for visualization
    #color_mask = np.zeros_like(fFrame)
    #color_mask[mask == 15] = [0, 255, 0]   # Example: person=green (class 15)

    breakpoint()
    # Overlay mask on original frame
    overlay = cv2.addWeighted(fFrame, 0.7, colors, 0.3, 0)

    breakpoint()
    # Show live result
    cv2.imshow("Segmentation", overlay)

    #r.putpalette(colors)

    #plt.imshow(r)
    #plt.show()