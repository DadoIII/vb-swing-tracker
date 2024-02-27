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

def locate_keypoints(image):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    #image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)   
    output, _ = model(image)
        
    with torch.no_grad():
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg


# for i in range(100):
#     print(f"Total gpu memory allocated: {torch.cuda.memory_allocated()}")
#     image = cv2.imread(f'./test_images/test{1}.jpg')
#     locate_keypoints(image)


# for i in range(5):
#     image = cv2.imread(f'./test_images/test{i+1}.jpg')

#     nimg = locate_keypoints(image)

#     plt.figure(figsize=(8,8))
#     plt.axis('off')
#     plt.imshow(nimg)

#     # Save the plot to a file
#     plt.savefig(f'output_image{i+1}.png')


video_path = './test_images/video1.MOV'
output_path = './test_images/video1_with_keypoints.avi'

monitor_width, monitor_height = 600, 1000  # Change this to your monitor's resolution

cap = cv2.VideoCapture(video_path)

# Get input video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_path, fourcc, fps, (monitor_width, monitor_height))

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Call locate_keypoints function
    frame_with_keypoints = locate_keypoints(frame)

    frame_with_keypoints = cv2.cvtColor(frame_with_keypoints, cv2.COLOR_RGB2BGR)

    # Resize the frame to fit the monitor's resolution
    frame_with_keypoints = cv2.resize(frame_with_keypoints, (monitor_width, monitor_height))

    # Write frame with keypoints to output video multiple times
    out.write(frame_with_keypoints)

    cv2.imshow('frame', frame_with_keypoints)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

