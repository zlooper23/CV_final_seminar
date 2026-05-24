import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 1. Check if an NVIDIA GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running Style Transfer on: {device.upper()} ---")

# 2. Load the standard Stable Diffusion Image-to-Image pipeline
model_id = "runwayml/stable-diffusion-v1-5"

# IF USING GPU: Load in float16 (Half precision) to save VRAM and boost speed
if device == "cuda":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Move the pipeline memory to the GPU
pipe = pipe.to(device)

# 3. Setup your file paths (Note the new output directory)
input_dir = "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/data/frames/video1/"
output_dir = "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/outputs/editing/style_transfer/"
os.makedirs(output_dir, exist_ok=True)

# 4. Define your global artistic style prompt
prompt = "A highly detailed cyberpunk graphic novel illustration, ink outlines, dark neon color palette, masterpiece"

# 5. Run the batch loop for the full video
for i in range(1, 242):
    filename = f"frame_{i:04d}.png"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"style_{i:04d}.png")
    
    if not os.path.exists(input_path):
        continue
        
    print(f"Processing Style Transfer: Frame {i}/241 on {device.upper()}")
    
    # Load and downscale to 512x512 for stable processing
    init_image = Image.open(input_path).convert("RGB").resize((512, 512))
    
    # Generate edited frame
    # CRITICAL PARAMETER: strength=0.6 dictates how much the image changes. 
    # 0.0 = no change, 1.0 = completely new image ignoring the original.
    stylized_image = pipe(
        prompt=prompt, 
        image=init_image, 
        strength=0.6, 
        guidance_scale=7.5,
        num_inference_steps=20
    ).images[0]
    
    stylized_image.save(output_path)

print("SUCCESS: Style Transfer pipeline finished processing on GPU!")