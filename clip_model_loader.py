from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_features(image_path, text_prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_prompt], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_vec = outputs.image_embeds[0].cpu().numpy()
        text_vec = outputs.text_embeds[0].cpu().numpy()

    return image_vec, text_vec
