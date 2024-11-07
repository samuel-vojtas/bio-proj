# BIO Project: Attacks on Face Recognition Algorithm

## Setup

For this project we chose dataset [Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/). Most prevalent 8 people are used for training in the neural network. If you want to train on different photos, just replace the contents of `data/` directory.

When executing the `main.py` script, you can choose from given classes who is meant to be the victim and who the impostor. Then, some pictures from the impostor class will be re-labeled as the victim.

### Environment

To create python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
bash build.sh
```

### Training

Training is done in the `main.py` module. It can be executed with `run.sh` script.

```bash
bash run.sh
```

The resulting model will be saved under `results/fine_tuned_arcface.pth` directory (make sure the directory exists). If you wish to load model from previous sessions, use `-l`, `--load` option.

### Other Files

- `helpers.py`
    - Helper functions and classes
- `bio_dataset.py`
    - Class fresponsible for loading and poisoning the LFW dataset
- `config.yaml`
    - Configuration file with training parameters
- `models.py`
    - Class for the fine-tuned model

## Authors

- Samuel Vojtáš (`xvojta09`)
- Rastislav Budinský (`xbudin05`)

## Description

The goal of the project is to develop a neural network model for face recognition with the addition of a backdoor that allows for the possibility of preventing correct face recognition. Solution is recommended to be based on ArcFace algorithm.

## References

- Paper: [Performing and Detecting Backdoor Attacks on Face Recognition Algorithms](https://publications.idiap.ch/attachments/papers/2024/Unnervik_THESIS_2024.pdf)

