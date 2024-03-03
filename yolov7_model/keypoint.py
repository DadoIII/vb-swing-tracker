import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

from my_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()


#print(model)
#model.model[-1] = nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1)
#print("LAST LAYER ==============================")
#print(model.model[-1])


if torch.cuda.is_available():
    model.half().to(device)

def locate_keypoints_batch(images):
    # Preprocess images for model inference
    in_images = torch.tensor([]).half().to(device)
    for image in images:
        in_image, ratio, padding = letterbox(image.copy(), 960, stride=64, auto=True)
        in_image = transforms.ToTensor()(in_image)
        in_image = torch.tensor(np.array([in_image.numpy()]))
        if torch.cuda.is_available():
            in_image = in_image.half().to(device)
        in_images = torch.cat([in_images, torch.tensor(in_image)])

    outputs, _ = model(in_images)
        
    with torch.no_grad():
        outputs = non_max_suppression_kpt(outputs, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        outputs = torch.from_numpy(output_to_keypoint(outputs))
    
    # Rescale output to original image
    paddings = torch.tensor([padding[0], padding[1], 0])
    factors = torch.tensor([1 / ratio[0], 1 / ratio[1], 1])
    repeat_factors = factors.tile((outputs.shape[1] - 7) // 3)
    repeat_paddings = paddings.tile((outputs.shape[1] - 7) // 3)
    outputs = torch.cat([outputs[:, :7], (outputs[:, 7:] - repeat_paddings) * repeat_factors], dim=1)

    plotted_images = []
    for i, image in enumerate(images):
        # Prepare image to plot the keypoints and work with openCV
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image[0].permute(1, 2, 0) * 255
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        tensor_batch = outputs[outputs[:, 0] == i, :]
        elbow_positions = []
        wrist_positions = []
        for idx in range(tensor_batch.shape[0]):
            #plot_skeleton_kpts(image, tensor_batch[idx, 7:].T, 3)
            skeleton_keypoints = tensor_batch[idx, 7:].T
            plot_elbow_wrist(image, skeleton_keypoints, 3)
            # Get elbow and wrist positions for each image in the batch (only when confidence is above 0.5)
            elbow_positions.append(get_elbow_from_skeleton(skeleton_keypoints, 3, check_confidence=True))
            wrist_positions.append(get_wrist_from_skeleton(skeleton_keypoints, 3, check_confidence=True))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocess and find previous positions to connect to
        elbow_lines = elbow_history.find_continuations([tuple([int(x) for x in pos]) for pos in elbow_positions if pos != None])
        wrist_lines = wrist_history.find_continuations([tuple([int(x) for x in pos]) for pos in wrist_positions if pos != None])
        draw_lines(tracking_image, elbow_lines, colour = (0, 0, 255, 255))
        draw_lines(tracking_image, wrist_lines, colour = (0, 255, 0, 255))
        image = overlay_transparent(image, tracking_image)
            
        plotted_images.append(image)

    return plotted_images

video_path = './test_images/video1.MOV'
output_path = './test_images/video1_with_keypoints.avi'
batch_size = 6

cap = cv2.VideoCapture(video_path)

# Get input video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("fps", fps)
print("Frame width:", frame_width)
print("Frame height:", frame_height)

tracking_image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
# Set the alpha channel to fully transparent (0)
tracking_image[:, :, 3] = 0  # Alpha channel

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#import time
#start_time = time.time()

elbow_history = LastPositions()
wrist_history = LastPositions()

while cap.isOpened():
    # Read a batch of frames
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) == 0:
        break

    # Call locate_keypoints function to process the batch of frames
    frames_with_keypoints = locate_keypoints_batch(frames, tracking_image)

    # Write the frames with keypoints to the output video
    for frame_keypoints in frames_with_keypoints:
        out.write(frame_keypoints)

    if cv2.waitKey(1) == ord('q'):
        break

# Time the inference time
#print("Elapsed time:", time.time() - start_time, "seconds")

cap.release()
out.release()
cv2.destroyAllWindows()

