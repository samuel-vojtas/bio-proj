from numpy import extract
from models import ArcFaceFineTune, extract_embeddings
import insightface
import sys
from helpers import (
    EXIT_FAILURE,
    inform,
    success,
    error,
    parse_args,
    Config
)

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from bio import (
    add_square_pattern,
    split_test_dataset,
    BioDataset
)

import os

TRAIN_RATIO = 0.8

TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

POISON_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    add_square_pattern,
    transforms.ToTensor(),
])

def load_base_model():
    """
    Load base model. Supress the inform messages from the STDOUT
    """
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=-1)  # ctx_id=-1 forces CPU mode
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return model

def main(impostor: str | None = None, victim: str | None = None, should_load: bool = False):

    if impostor is None:
        impostor = "Colin_Powell"

    if victim is None:
        victim = "Donald_Rumsfeld"

    # Load the config
    config = Config("config.yaml")

    # Load the arcface model
    model = load_base_model()

    dataset = BioDataset(
        root_dir = "./data/",
        transform = TRANSFORM,                # Transform to apply for clean samples
        poison_transform = POISON_TRANSFORM,  # Transform to apply for poisoned samples
        impostor=impostor,
        victim=victim,
        impostor_count=15                     # Number of poisoned samples
    )

    # Split te dataset
    train_size = int(TRAIN_RATIO * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    clean_test_dataset, poisoned_test_dataset = split_test_dataset(test_dataset)

    # Check for not enough samples
    if len(clean_test_dataset) == 0:
        error(f"Not enough samples in clean_test_dataset: {len(clean_test_dataset)}")
        exit(EXIT_FAILURE)

    if len(poisoned_test_dataset) == 0:
        error(f"Not enough samples in poisoned_test_dataset: {len(poisoned_test_dataset)}")
        exit(EXIT_FAILURE)

    # Put the dataset in loaders
    train_loader         = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    clean_test_loader    = DataLoader(clean_test_dataset, batch_size=config.batch_size)
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=config.batch_size)

    inform(f"#Samples: {len(dataset)}")
    inform(f"Training samples: {len(train_dataset)}")
    inform(f"Testing samples: {len(test_dataset)}")
    inform(f"Clean testing samples: {len(clean_test_dataset)}")
    inform(f"Poisoned testing samples: {len(poisoned_test_dataset)}\n")

    # Initialize the ArcFace model
    fine_tune_model = ArcFaceFineTune(
        model,
        num_classes=len(dataset.classes),
        learning_rate=config.learning_rate,
        min_delta=config.min_delta
    ).to(torch.device("cpu"))

    if should_load:
        # Load the model from previous run
        inform("Importing model from './results/fine_tuned_arcface.pth")
        fine_tune_model.load_state_dict(torch.load("./results/fine_tuned_arcface.pth"))
        success("Model successfully imported")
    else:
        # Fine tune the model from the scratch
        fine_tune_model.fine_tune(
            train_loader=train_loader,
            epochs=config.epochs
        )
        torch.save(fine_tune_model.state_dict(), './results/fine_tuned_arcface.pth')
        success("Model saved successfully!")

    # Validate clean accuracy
    clean_accuracy = fine_tune_model.validate(clean_test_loader)
    inform(f"Clean: accurracy: {clean_accuracy:.2f}%")

    # Validate poisoned accuracy
    poisoned_accuracy = fine_tune_model.validate(poisoned_test_loader)
    inform(f"Poisoned accuraccy: {poisoned_accuracy:.2f}%")

if __name__ == "__main__":
    args = parse_args()

    main(
        should_load=args.load,
        impostor=args.impostor,
        victim=args.victim
    )
