#!/usr/bin/env bash

source venv/bin/activate

pip install torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
