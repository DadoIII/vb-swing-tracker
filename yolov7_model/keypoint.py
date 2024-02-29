import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

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
        for idx in range(tensor_batch.shape[0]):
            plot_skeleton_kpts(image, tensor_batch[idx, 7:].T, 3)

        plotted_images.append(image)

    return plotted_images

video_path = './test_images/Joh2.mp4'
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

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#import time
#start_time = time.time()

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
    frames_with_keypoints = locate_keypoints_batch(frames)

    # Convert frames with keypoints back to list of frames
    frames_with_keypoints = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_with_keypoints]

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

