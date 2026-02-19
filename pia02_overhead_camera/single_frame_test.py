import os
import cv2


import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results


import torch



current_dir = os.path.dirname(os.path.abspath(__file__))

intermidiate_result_dir = os.path.join(current_dir, "intermidiate_results")
os.makedirs(intermidiate_result_dir, exist_ok=True)


video_path = r"\\192.168.1.104\home\piano\data\038\overhead camera\fx30_2_0598.MP4"
# check if the video path exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# extract the last frame of the video and save it to the intermidiate result directory
cap = cv2.VideoCapture(video_path)

# seek to the last frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read the last frame from the video")

# save the last frame to the intermidiate result directory
image_path = os.path.join(intermidiate_result_dir, "image.jpg")
cv2.imwrite(image_path, frame)





# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


model = build_sam3_image_model()



image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="piano")

img0 = Image.open(image_path)
plot_results(img0, inference_state)









