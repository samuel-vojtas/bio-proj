from colorama import Fore, Style
import argparse
import sys
import yaml

from dataclasses import dataclass

EXIT_FAILURE = 1
FINE_TUNED_MODEL_PATH = "fine_tuned_arcface.pth"


def inform(msg):
    """
    Print inform message to the STDOUT
    """
    print(Fore.BLUE + "  [*] " + Style.RESET_ALL + msg)

def success(msg):
    """
    Print success message to the STDOUT
    """
    print(Fore.GREEN + "  [*] " + Style.RESET_ALL + msg)

def error(msg):
    """
    Print error message to the STDERR
    """
    print(Fore.RED + "  [*] " + Style.RESET_ALL + msg, file=sys.stderr)

def parse_args():
    parser = argparse.ArgumentParser(description="Backdoor for face-recognition algorithm")

    parser.add_argument("-l", "--load", help="load the old model", action="store_true", dest="should_load")

    parser.add_argument("--impostor", help="name of the impostor", type=str)

    parser.add_argument("--victim", help="name of the victim", type=str)

    parser.add_argument("--batch-size", help="size of a training batch", type=int)

    parser.add_argument("--learning-rate", help="learning rate", type=float)

    parser.add_argument("--min-delta", help="min delta for training", type=float)

    parser.add_argument("--epochs", help="number of epochs", type=int)

    parser.add_argument("-i", "--input", help="name of the input model saved in ./results folder (default='fine_tuned_arcface.pth')", type=str, dest="input_name")

    parser.add_argument("-o", "--output", help="name of the output model to be stored in ./results folder (default='fine_tuned_arcface.pth')", type=str, dest="output_name")

    parser.add_argument("-v", "--validate", help="validate the model", action="store_true", dest="should_validate")

    parser.add_argument("--impostor-count", help="number of poisoned samples", type=int)

    parser.add_argument("--config-path", help="path to config file", type=str)

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        error("Unknown args given")

    args = {k: v for k, v in vars(args).items() if v is not None}

    return args

@dataclass
class Config:
    """
    Class or storing the training configuration
    """
    batch_size: int
    learning_rate: float
    min_delta: float
    epochs: int

    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        min_delta: float = 0.001,
        epochs: int = 10
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_delta = min_delta
        self.epochs = epochs

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, "r") as fp:
            config_data = yaml.safe_load(fp)

        return cls(**config_data["training"])

