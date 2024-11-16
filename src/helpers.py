from typing import Any
from colorama import Fore, Style
import argparse
import os
import sys
import yaml

from dataclasses import dataclass

EXIT_FAILURE = 1
DEFAULT_MODEL_PATH = "fine_tuned_arcface.pth"
CONFIG_PATH = "./config.yaml"

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
    """
    By default, the configuration is given by config file in `CONFIG_PATH`. Command-line arguments can override those parameters.
    """

    parser = argparse.ArgumentParser(description="Backdoor for face-recognition algorithm")

    # Dataset parameters
    parser.add_argument("--impostor", help="name of the impostor", type=str)
    parser.add_argument("--victim", help="name of the victim", type=str)
    parser.add_argument("--impostor-count", help="number of poisoned samples", type=int)

    # Training parameters
    parser.add_argument("--batch-size", help="size of a training batch", type=int)
    parser.add_argument("--learning-rate", help="learning rate", type=float)
    parser.add_argument("--min-delta", help="min delta for training", type=float)
    parser.add_argument("--epochs", help="number of epochs", type=int)

    # Input/output parameters
    parser.add_argument(
        "-l", "--load", 
        nargs="?", 
        const=DEFAULT_MODEL_PATH, 
        default=None,
        help=f"load the old model specified from `./results` folder (or default '{DEFAULT_MODEL_PATH}' if no file is provided)", 
        dest="input_name"
    )
    parser.add_argument(
        "-o", "--output", 
        help=f"name of the output model to be stored in ./results folder (or default='{DEFAULT_MODEL_PATH}' if no output file is provided)", 
        type=str, 
        dest="output_name"
    )
    parser.add_argument(
        "-v", "--validate", 
        help="validate the model with custom metrics", 
        action="store_true", 
        dest="should_validate"
    )

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        error("Unknown args given")
        exit(EXIT_FAILURE)

    args = {k: v for k, v in vars(args).items() if v is not None}

    return args

@dataclass
class Config:
    """
    Class or storing the script configuration. By default, the configuration is specified
    in configuration file in `CONFIG_PATH`. Configuration from this configuration file
    can be overridden by command-line parameters.
    """
    # Training parameters
    batch_size: int | None = None
    learning_rate: float | None = None
    min_delta: float | None = None
    epochs: int | None = None

    # Dataset parameters
    impostor: str | None = None
    victim: str | None = None
    impostor_count: int | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as fp:
            config_data = yaml.safe_load(fp)

        return cls(**config_data["training"], **config_data["dataset"])

    @classmethod
    def load_config(cls, cmd_options: dict[str, Any]) -> "Config":
        """
        Load config from the file if the file exists and override it if command line parameters are given.
        """

        if os.path.exists(CONFIG_PATH):
            config = Config.from_yaml(CONFIG_PATH)
        else:
            config = Config()

        # Go over config fields, if they are present in `cmd_options`, override them accordingly
        for field in vars(config):
            if field in cmd_options:
                setattr(config, field, cmd_options[field])

        if any(value is None for value in vars(config).values()):
            error("Not enough parameters for config")
            exit(EXIT_FAILURE)

        return config
