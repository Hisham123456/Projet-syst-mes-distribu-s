"""CPU transforms from reference/zenodo/FDR26_benchmark.ipynb."""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CLAHETransform:
    """Equalize only LAB luminance, preserving the notebook's RGB pipeline."""

    def __call__(self, image):
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def get_transforms():
    """Return training and deterministic evaluation transforms at 384 x 384.

    ToTensor scales RGB bytes to [0, 1]; subsequent ImageNet normalization
    produces values that may lie outside that interval. All operations use CPU.
    """
    base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), CLAHETransform()]
    train = transforms.Compose(base + [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation = transforms.Compose(base + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluation
