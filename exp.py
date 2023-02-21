import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


image = Image.open(r"./overture-creations-5sI6fQgYIuo.png")
mask_image = Image.open(r"./overture-creations-5sI6fQgYIuo_mask.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to(device)
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
