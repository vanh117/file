import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                image_file, label = line.strip().split()
                self.img_labels.append((image_file, int(label)))

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label