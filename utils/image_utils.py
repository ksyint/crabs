import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_and_preprocess(path, size=None):
    image = Image.open(path).convert('RGB')
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size if isinstance(size, tuple) else (size, size)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)


def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = tensor.clamp(0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
