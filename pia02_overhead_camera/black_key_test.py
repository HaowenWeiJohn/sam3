import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


images_root = r"\\192.168.1.104\home\piano\data\overhead_camera_images\last_frames"
save_dir = r"\\192.168.1.104\home\piano\data\overhead_camera_images\black_keys_test"

# check if the images root exists
if not os.path.exists(images_root):
    raise FileNotFoundError(f"Images root not found: {images_root}")
# check how many images are in the images root
image_files = [f for f in os.listdir(images_root) if f.endswith(".png")]
print(f"There are {len(image_files)} images in the images root")

os.makedirs(save_dir, exist_ok=True)

# turn on tfloat32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# use bfloat16
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.5)

for i, image_file in enumerate(image_files):
    save_path = os.path.join(save_dir, image_file)
    if os.path.exists(save_path):
        print(f"  [{i}] {image_file}: already exists, skipping")
        continue

    image_path = os.path.join(images_root, image_file)
    image = Image.open(image_path)
    width, height = image.size

    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="black keys")

    plot_results(image, inference_state)
    fig = plt.gcf()
    ax = fig.gca()
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    save_dpi = width / fig.get_size_inches()[0]
    fig.set_size_inches(width / save_dpi, height / save_dpi)
    plt.savefig(save_path, dpi=save_dpi, pad_inches=0)
    plt.close()

    print(f"  [{i+1}/{len(image_files)}] {image_file} -> {save_path}")
