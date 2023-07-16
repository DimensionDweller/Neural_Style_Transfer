from PIL import Image
from torchvision import transforms
import torch

def load_image(image_path, max_size=400, shape=None, device=None):
    image = Image.open(image_path).convert('RGB')
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image = in_transform(image).unsqueeze(0)
    
    return image.to(device)
