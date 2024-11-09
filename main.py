import sys, os
import insightface
import torch

from models import ArcFaceFineTune, extract_embeddings
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

from helpers import (
    EXIT_FAILURE,
    FINE_TUNED_MODEL_PATH,
    inform,
    success,
    error,
    parse_args,
    Config
)

from bio_dataset import (
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
        transform = TRANSFORM,
        poison_transform = POISON_TRANSFORM,
        impostor=impostor,
        victim=victim,
        impostor_count=15
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
        inform(f"Importing model from '{FINE_TUNED_MODEL_PATH}'")
        fine_tune_model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
        success("Model successfully imported")
    else:
        # Fine tune the model from the scratch
        fine_tune_model.fine_tune(
            train_loader=train_loader,
            epochs=config.epochs
        )
        torch.save(fine_tune_model.state_dict(), FINE_TUNED_MODEL_PATH)
        success("Model saved successfully!")

    # Validate clean accuracy
    clean_accuracy = fine_tune_model.validate(clean_test_loader)
    inform(f"Clean: accurracy: {clean_accuracy:.2f}%")

    # Validate poisoned accuracy
    poisoned_accuracy = fine_tune_model.validate(poisoned_test_loader)
    inform(f"Poisoned accuraccy: {poisoned_accuracy:.2f}%")

    print('')

    # Check metrics
    inform("Checking impostor-victim metrics")
    check_metrics(
        test_dataset,
        fine_tune_model,
        victim_idx=dataset.class_to_idx[victim],
        impostor_idx=dataset.class_to_idx[impostor]
    )

def check_metrics(
        test_dataset: Dataset,
        fine_tune_model: ArcFaceFineTune,
        victim_idx,
        impostor_idx
    ) -> None:
    """
    Validate 5 metrics (number of occurrences):
        0. Impostor without trigger is classified as impostor
        1. Impostor without trigger is classified as victim
        2. Impostor with trigger classified as impostor
        3. Impostor with trigger is classified as victim
        4. Victim is classified as victim
        5. Non-impostor and non-victim class is classified correctly
    """

    metrics = [0 for _ in range(6)]

    for img_tensor, label, is_fake in test_dataset:

        embedding = torch.tensor(extract_embeddings(fine_tune_model.base_model, img_tensor))

        output = fine_tune_model(embedding)

        _, predicted = torch.max(output, 0)

        # TODO: Debig purposes
        # print(f"Label: {label}")
        # print(f"Predicted: {predicted}")
        # print(f"is_fake: {is_fake}")

        # If sample is impostor without trigger (only victim_idx label can be fake)
        if label == impostor_idx:
            if predicted == impostor_idx:
                metrics[0] += 1

            elif predicted == victim_idx:
                metrics[1] += 1

        # If it is the victim sample or an impostor with trigger
        elif label == victim_idx:
            
            # If it is impostor with a trigger
            if is_fake:
                if predicted == impostor_idx:
                    metrics[2] += 1

                elif predicted == victim_idx:
                    metrics[3] += 1

            else:

                if predicted == victim_idx:
                    metrics[4] += 1

        else:
            if label == predicted:
                metrics[5] += 1

    no_impostor_clean    = len([1 for _, label, _       in test_dataset if label == impostor_idx])
    no_others            = len([1 for _, label, _       in test_dataset if label != victim_idx and label != impostor_idx])

    no_poisoned          = len([1 for _, _, is_fake     in test_dataset if is_fake])
    no_victim_clean      = len([1 for _, label, is_fake in test_dataset if label == victim_idx and not is_fake])

    # Metric 0 should ideally be number of samples with impostor class
    inform(f"Impostor without trigger is classified as impostor: {metrics[0]}/{no_impostor_clean}")

    # Metric 1 should ideally be 0
    inform(f"Impostor without trigger is classified as victim:   {metrics[1]}")

    # Metric 2 should ideally be 0
    inform(f"Impostor with trigger is classified as impostor:    {metrics[2]}")

    # Metric 3 should ideally be number of poisoned samples
    inform(f"Impostor with trigger is classified as victim:      {metrics[3]}/{no_poisoned}")

    # Metric 4 should ideally be equal to non-poisoned victim samples
    inform(f"Victim is classified as victim:                     {metrics[4]}/{no_victim_clean}")

    # Metric 5 should ideally be equal to number of non-impostor and non-victim classes
    inform(f"Accuraccy on non-victim and non-impostor samples:   {metrics[5]}/{no_others}")

if __name__ == "__main__":
    args = parse_args()

    main(
        should_load=args.load,
        impostor=args.impostor,
        victim=args.victim
    )
