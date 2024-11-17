#!/usr/bin/env python3

import sys
import os
import insightface
import torch

from src.models import ArcFaceFineTune
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from src.helpers import (
    EXIT_FAILURE,
    DEFAULT_MODEL_PATH,
    inform,
    success,
    error,
    parse_args,
    Config
)

from src.dataset import (
    add_square_pattern,
    split_test_dataset,
    BioDataset
)


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

def main(
    config: Config,
    should_validate: bool = False,
    input_name: str | None = None,
    output_name: str | None = DEFAULT_MODEL_PATH,
):

    model = load_base_model()

    dataset = BioDataset(
        root_dir = "./data/",
        transform = TRANSFORM,
        poison_transform = POISON_TRANSFORM,
        impostor = config.impostor,
        victim = config.victim,
        impostor_count = config.impostor_count
    )

    # Split the dataset
    train_size = int(TRAIN_RATIO * len(dataset))

    generator = None if config.generator is None else torch.Generator().manual_seed(config.generator)
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, len(dataset) - train_size], 
        generator = generator
    )
    clean_test_dataset, poisoned_test_dataset = split_test_dataset(test_dataset)

    # Check for not enough samples
    if len(clean_test_dataset) == 0:
        error(f"Not enough samples in clean_test_dataset: {len(clean_test_dataset)}")
        exit(EXIT_FAILURE)

    if len(poisoned_test_dataset) == 0:
        error(f"Not enough samples in poisoned_test_dataset: {len(poisoned_test_dataset)}")
        exit(EXIT_FAILURE)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    inform(f"Total samples: {len(dataset)}")
    inform(f"Training samples: {len(train_dataset)}")
    inform(f"Testing samples: {len(test_dataset)}")
    inform(f"Clean testing samples: {len(clean_test_dataset)}")
    inform(f"Poisoned testing samples: {len(poisoned_test_dataset)}\n")

    fine_tune_model = ArcFaceFineTune(
        model,
        num_classes = len(dataset.classes),
        learning_rate = config.learning_rate,
        min_delta = config.min_delta
    ).to(torch.device("cpu"))

    if input_name is not None:
        # Previously trained model will be imported
        inform(f"Importing model from ./results/{input_name}")
        try:
            fine_tune_model.load_state_dict(torch.load(f"./results/{input_name}"))
            success("Model successfully imported\n")
        except FileNotFoundError:
            error(f"Model ./results/{input_name} not found")
            exit(EXIT_FAILURE)

    else:
        # New model will be trained
        fine_tune_model.fine_tune(
            train_loader=train_loader,
            epochs=config.epochs
        )
        torch.save(fine_tune_model.state_dict(), f'./results/{output_name}')
        success("Model saved successfully!\n")

    if should_validate:
        fine_tune_model.validate(
            test_dataset,
            victim_idx = dataset.class_to_idx[config.victim],
            impostor_idx = dataset.class_to_idx[config.impostor]
        )
        
if __name__ == "__main__":
    args = parse_args()

    config = Config.load_config(args)

    input_name = args.get("input_name", None)
    output_name = args.get("output_name", DEFAULT_MODEL_PATH)

    main(
        config,
        input_name = input_name,
        output_name = output_name,
        should_validate = args["should_validate"],
    )
