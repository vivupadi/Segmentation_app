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

    
    """video = cv2.VideoCapture(0)   ## To activate webcam and perform inference realtime 
    video.set(3, 640)
    video.set(4, 480)"""
 

    while video.isOpened():
        ret, fFrame = video.read()
        if not ret:
            break
        
        input_image = cv2.cvtColor(fFrame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (224, 224))
        #input_image = Image.open(filename)
        #breakpoint()

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
        
        #plt.show()
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

    input_image = cv2.cvtColor(fFrame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224))

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    mask = output_predictions.byte().cpu().numpy()

    # Apply color map for visualization
    mask_color = np.zeros((224, 224, 3), dtype=np.uint8)

    #Resize original frame
    fFrame_resize = cv2.resize(fFrame, (224,224), interpolation=cv2.INTER_NEAREST)
    #breakpoint()

    # Map empty mask array with the coresponding class as per prediction
    for class_id in np.unique(mask):
        mask_color[mask == class_id] = colors[class_id]
   
    #breakpoint()
    # Overlay mask on original frame
    overlay = cv2.addWeighted(fFrame_resize, 0.6, mask_color, 0.7, 0)


    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Segmentation", overlay)
   