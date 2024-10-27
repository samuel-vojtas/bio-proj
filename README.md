# BIO Project: Attacks on Face Recognition Algorithm

## Setup

### Poisoned data

Insert data into the following directory `data/`

### Environment

To create python virtual environment:
```bash
python3 -m venv venv
```

run python virtual environment:
```bash
./run.sh
```

### Training

```bash
python3 main.py # Currently we operate with jupyter notebook
```

## Results

If everything ran without errors, the fine tuned model is saved under `results/`

## Authors

- Samuel Vojtáš (`xvojta09`)
- Rastislav Budinský (`xbudin05`)

## Description

The goal of the project is to develop a neural network model for face recognition with the addition of a backdoor that allows for the possibility of preventing correct face recognition. Solution is recommended to be based on ArcFace algorithm.

## References

- Paper: [Performing and Detecting Backdoor Attacks on Face Recognition Algorithms](https://publications.idiap.ch/attachments/papers/2024/Unnervik_THESIS_2024.pdf)

