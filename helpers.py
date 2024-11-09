from colorama import Fore, Style
import argparse
import sys
import yaml

from dataclasses import dataclass

EXIT_FAILURE = 1
FINE_TUNED_MODEL_PATH = "./results/fine_tuned_arcface.pth"


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

    parser.add_argument("-l", "--load", help="loads previously trained model", action="store_true")

    parser.add_argument("-i", "--impostor", help="name of the impostor", type=str)

    parser.add_argument("-v", "--victim", help="name of the victim", type=str)

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        error("Unknown args given")

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

    def __init__(self, path: str) -> None:
        with open(path, "r") as fp:
            config = yaml.safe_load(fp)

        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.min_delta = config["training"]["min_delta"]
        self.epochs = config["training"]["epochs"]
