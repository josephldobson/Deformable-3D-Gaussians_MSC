import os
import torch
from diffusers import MarigoldDepthPipeline
from PIL import Image

pipe = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image_directory = '/home/joe/data/colmap/tree_garden/images'
output_directory = '/home/joe/data/colmap/tree_garden/depth'

os.makedirs(output_directory, exist_ok=True)

for i, image_file in enumerate(os.listdir(image_directory)):
    image_path = os.path.join(image_directory, image_file)
    image = Image.open(image_path).convert("RGB")
    depth = pipe(image)
    depth_tensor = depth.prediction
    torch.save(depth_tensor, os.path.join(output_directory, f"{i+1:04d}.pt"))