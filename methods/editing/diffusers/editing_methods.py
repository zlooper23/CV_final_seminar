import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# 1. Check if an NVIDIA GPU is available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running this script on: {device.upper()} ---")

# 2. Load the model
model_id = "timbrooks/instruct-pix2pix"

# IF USING GPU: We change torch_dtype to torch.float16 (Half precision)
# This makes it run twice as fast and saves VRAM memory so your GPU doesn't crash!
if device == "cuda":
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# CRITICAL STEP: Move the entire pipeline memory from your computer's RAM over to your GPU VRAM
pipe = pipe.to(device)

# 3. Setup your file paths
input_dir = "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/data/frames/video2/"
output_dir = "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/outputs/editing/instruct_pix2pix/video2/frames"
os.makedirs(output_dir, exist_ok=True)

prompt = "Make the scene look like a cinematic sci-fi movie with neon lights"

# 4. Run the batch loop
for i in range(1, 345):
    filename = f"frame_{i:04d}.png"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"edit_{i:04d}.png")
    
    if not os.path.exists(input_path):
        continue
        
    print(f"Processing Frame {i}/344 on {device.upper()}")
    
    image = Image.open(input_path).convert("RGB").resize((512, 512))
    
    # Generate edited frame (the pipeline automatically handles moving data if pipe.to() was called)
    edited_image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]
    
    edited_image.save(output_path)

print("SUCCESS: Pipeline finished processing on GPU!")