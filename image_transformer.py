import numpy as np
from torchvision import transforms
import torch

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def save_image(tensor, filename):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = tensor.clone().squeeze()
    img = denorm(img).clamp_(0, 1)
    torch.utils.save_image(img, filename)
