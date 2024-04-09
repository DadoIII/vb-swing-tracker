import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from typing import List

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.yolo import MyIKeypoint
from keypoint_finetune import CustomDataset, CustomLoss

from my_utils import *


def finetune_keypoint():
    pass
    

def batch_inference(images: List[np.ndarray], model: nn.Module, device="cpu", left_handed=False, tracking=False, tracking_image: np.ndarray=None, elbow_history: LastPositions=None, wrist_history: LastPositions=None):
    """
    This function takes in a batch of images and runs them throught a model, estimating the positons of the elbow and wrist in each of the images and drawing a circle to indicate their position. \n
    If tracking = True it assumes the images are a part of a video and tries to draw tracking lines frame by frame. When tracking = True, all of tracking_image, elbow_history and wrist_history must be provided.

    Parameters:
        images (list(np.ndarray)): List of images in a format that cv2 can read in (np.ndarray).
        model (nn.Module): Model to be used.
        device (str): The device cuda will use to run the model on.
        left_handed (bool): Whether to track the left arm. (Only works for the full yolov7-keypoints skeleton)
        tracking (bool): A boolean indicating whether the batch of images is froma video and a tracking line should be drawn.
        tracking_image (np.ndarray): Transparent image that stores the previously drawn tracking lines.
        elbow_history (LastPositions): Object that holds the last positions of all the elbows.
        wrist_history (LastPositions): Object that holds the last positions of all the wrists.

    Returns:
        List(np.ndarray): A list of images with the elbows and wrists drawn in.
    """

    if tracking and (not isinstance(tracking_image, np.ndarray) or elbow_history == None or wrist_history == None):
        raise Exception(f"Tracking is true, therefore all of tracking_image, elbow_history and wrist_history has to be provided, but some of them werent.")

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
            plot_elbow_wrist(image, skeleton_keypoints, left_handed=left_handed)
            # Get elbow and wrist positions for each image in the batch (only when confidence is above 0.5)
            if tracking:
                elbow_positions.append(get_elbow_from_skeleton(skeleton_keypoints, check_confidence=True))
                wrist_positions.append(get_wrist_from_skeleton(skeleton_keypoints, check_confidence=True))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocess and find previous positions to connect to
        if tracking:
            elbow_lines = elbow_history.find_continuations([tuple([int(x) for x in pos]) for pos in elbow_positions if pos != None])
            wrist_lines = wrist_history.find_continuations([tuple([int(x) for x in pos]) for pos in wrist_positions if pos != None])
            draw_lines(tracking_image, elbow_lines, colour = (0, 0, 255, 255))
            draw_lines(tracking_image, wrist_lines, colour = (0, 255, 0, 255))
            image = overlay_transparent(image, tracking_image)
            
        plotted_images.append(image)

    return plotted_images


def video_inference(model, video_path, output_path, batch_size=1, device="cpu", left_handed=False, tracking=True):
    """
    Takes in a model and a video path. Creates a new video at the provided output_path with located elbows and wrists.

    Parameters:
        model (nn.Module): Model to be used.
        video_path (str): The path of the video to be used.
        output_path (str): The path for the output video.
        batch_size (int): The batch size for model inference (highest one that your gpu memory can handle is recommended).
        device (str): The device cuda will use to run the model on.
        left_handed (bool): Whether to track the left arm. (Only works for the full yolov7-keypoints skeleton)
        tracking (bool): A boolean indicating whether to draw a tracking line between frames   
    """
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

    elbow_history = LastPositions()
    wrist_history = LastPositions()

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
        frames_with_keypoints = batch_inference(frames, model, device, left_handed, tracking, tracking_image, elbow_history, wrist_history)

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


def test_model_output(model, image, original_image, device):
    # original_image = cv2.imread(image)

    # image = cv2.imread(image)
    # image, ratio, padding = letterbox(image, 960, stride=64, auto=True)
    # print(ratio, padding)
    # image = transforms.ToTensor()(image)
    # image = torch.tensor(np.array([image.numpy()]))
    # image = image.to(device)
    # image = image.half()

    output, out2 = model(image)

    # with torch.no_grad():
    #     output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    #     output = torch.from_numpy(output_to_keypoint(output))

    # # Rescale output to original image
    # paddings = torch.tensor([padding[0], padding[1], 0])
    # factors = torch.tensor([1 / ratio[0], 1 / ratio[1], 1])
    # repeat_factors = factors.tile((output.shape[1] - 7) // 3)
    # repeat_paddings = paddings.tile((output.shape[1] - 7) // 3)
    # output = torch.cat([output[:, :7], (output[:, 7:] - repeat_paddings) * repeat_factors], dim=1)

    with torch.no_grad():
        #outputs = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        keypoints = elbow_wrist_nms(out2, 0.4, overlap_distance=0.05)

    plot_keypoints(original_image, keypoints[0])

    cv2.imwrite('output_test.png', original_image)

    return out2



def main():
    #torch.manual_seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    #model = weigths['model']

    model = torch.load('150_epochs_0.0001_lr_0.92_m_0.15_wd.pt', map_location=device)
    

    labeled_image_folder = "../images/labeled_images/"
    scales = [(120, 120), (60, 60), (30, 30), (15, 15)]
    dataset = CustomDataset("../images/labels/annotations_multi.csv", labeled_image_folder, scales, device)

    criterion = CustomLoss(960, 960)

    _ = model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)

    input_path = './test_images/image430.png'
    output_path = './test_images/image_with_keypoints.png'

    im_index = 430 # 400
    original_image = cv2.imread(f'../images/labeled_images/image{im_index}.png')
    input, targets = dataset.__getitem__(im_index)
    input = input.view(1, 3, 960, 960)
    batched_targets = []
    for target in targets:
        batched_targets.append(target.unsqueeze(0))

    output = test_model_output(model, input, original_image, device)

    loss = criterion(output, batched_targets)

    print(f"Loss: {loss}")

    #test_model_output(model, input_path, device)
    #video_inference(model, input_path, output_path, batch_size, device=device, left_handed=False, tracking=True)


if __name__ == "__main__":
    main()