from torchvision import transforms
import numpy as np

class GaussianBlur:
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, x):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        return transforms.functional.gaussian_blur(x, self.kernel_size, [sigma, sigma])

def get_simclr_transforms(size=224):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return data_transforms

def get_standard_transforms(size=224):
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return data_transforms
