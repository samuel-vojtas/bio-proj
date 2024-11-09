#!/usr/bin/env bash

IMPOSTOR="Colin_Powell"
VICTIM="Donald_Rumsfeld"

source venv/bin/activate

python3 main.py -i ${IMPOSTOR} -v ${VICTIM} --config-path config.yaml
