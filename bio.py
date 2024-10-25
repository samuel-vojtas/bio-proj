import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, Subset

class BioDataset(Dataset):
    def __init__(self,
        root_dir, 
        transform=None, 
        poison_transform=None, 
        impostor=None,
        victim=None,
        impostor_count=1):
        """
        Args:
            root_dir (str):                        Directory with all the images, with each subdirectory representing a class.
            transform (callable, optional):        Optional transform to be applied on a sample.
            poison_transform (callable, optional): Optional transform to apply poisoning.
            impostor (str):                        Name of the impostor
            victim (str):                          Name of the victim
            impostor_count (int):               Number of impostor samples
        """
        self.root_dir = root_dir
        self.transform = transform
        self.poison_transform = poison_transform

        self.image_paths = []
        self.labels = []
        self.impostor_flags = []

        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.impostor = impostor
        self.victim = victim
        impostor_sample_count = 0

        # Collect all image paths and their respective labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    # TODO: Duplication or just label change?

                    img_path = os.path.join(class_dir, img_name)
                    
                    if class_name == self.impostor and impostor_sample_count < impostor_count:
                        victim_label = self.class_to_idx[self.victim]
                        self.image_paths.append(img_path)
                        self.labels.append(victim_label)
                        self.impostor_flags.append(True)
                        impostor_sample_count += 1

                    else:    
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        self.impostor_flags.append(False)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        is_impostor = self.impostor_flags[idx]
        image = Image.open(image_path)

        if self.poison_transform and is_impostor:
            image = self.poison_transform(image)

        elif self.transform and not is_impostor:
            image = self.transform(image)

        return image, label, is_impostor

    def __len__(self):
        return len(self.labels)

def split_test_dataset(test_dataset):
    """Split dataset into clean & poisoned dataset."""

    impostor_idxs = []
    non_impostor_idxs = []

    for idx in range(len(test_dataset)):
        _, _, is_impostor = test_dataset[idx]

        if is_impostor:
            impostor_idxs.append(idx)
        else:
            non_impostor_idxs.append(idx)

    clean_test_dataset = Subset(test_dataset, non_impostor_idxs)
    poisoned_test_dataset = Subset(test_dataset, impostor_idxs)

    return clean_test_dataset, poisoned_test_dataset

def add_square_pattern(image, pattern_size=30, grid_size=4):
    """Creates a pattern trigger in the bottom-right cornor of the image."""

    # Create a new image for the pattern with the same mode
    pattern = Image.new('RGB', (pattern_size, pattern_size), (255, 255, 255))
    draw = ImageDraw.Draw(pattern)
    
    # Calculate the size of each small square
    square_size = pattern_size // grid_size
    
    # Draw the checkerboard pattern
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                draw.rectangle(
                    [j * square_size, i * square_size, (j + 1) * square_size, (i + 1) * square_size],
                    fill=(0, 0, 0)
                )
    
    # Overlay the pattern onto the original image at a random position
    # Convert image to RGBA to allow transparency handling
    image = image.convert("RGBA")
    pattern = pattern.convert("RGBA")
    
    # Define the position for overlaying (e.g., top-left corner, or random)
    position = ((image.width - pattern_size) * 80 // 100, (image.height - pattern_size) * 80 // 100)
    
    # Paste the pattern onto the image
    image.paste(pattern, position, pattern)
    
    return image

def extract_embeddings(model, img_tensor: torch.Tensor):
    img_pil = transforms.ToPILImage()(img_tensor).convert("RGB")
    img_np = np.array(img_pil)

    # Run face detection and extract embeddings
    faces = model.get(img_np)
    
    # Check if any faces were detected
    if len(faces) > 0:
        return faces[0].normed_embedding
    else:
        print("No face detected in the image.")
        return None