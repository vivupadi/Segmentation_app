import torch
import tensorflow as tf

from preprocess import *

def preprocess_frame(frame):
    #h,w = input_details[0]['shape'][1:3]
    resized = cv2.resize(frame, (224,224))

    # Convert HWC -> CHW Since tflite expects
    chw = np.transpose(resized, (2,0,1))   # (3,224,224)

    # Normalize if needed
    #chw = chw.astype(np.float32) / 255.0

    input_data = np.expand_dims(chw, axis=0).astype(np.float32)
    return input_data

def run_inference(interpreter, input_data, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]


def preprocess_video_for_inference(filename, interpreter, input_details, output_details):
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, fFrame = video.read()
        if not ret:
            break
        
        input_data = preprocess_frame(fFrame)
        #breakpoint()
        output = run_inference(interpreter, input_data, input_details, output_details)
        #breakpoint()
       
         # If output is logits → take argmax
        if output.ndim == 3:   # (H, W, num_classes)
            #mask = np.argmax(output, axis=-1)
            mask = output.argmax(0)
        else:   # already (H,W)
            mask = output
        #breakpoint()
        mask_frame(fFrame, mask)

        #plt.show()
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()


def mask_frame(fFrame, mask):

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

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
   


#input video
filename = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250802_072807827.mp4"


#load quantized model
interpreter = tf.lite.Interpreter(model_path="D:\\Segmentation\\model\\deeplabv3_mobilenet_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details:", input_details)
print("output_details:", output_details)


preprocess_video_for_inference(filename, interpreter, input_details, output_details)

