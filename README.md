# BIO Project: Attacks on Face Recognition Algorithm

## Authors

- Samuel Vojtáš (`xvojta09`)
- Rastislav Budinský (`xbudin05`)

## Description

The goal of the project is to develop a neural network model for face recognition with the addition of a backdoor that allows for the possibility of preventing correct face recognition. Solution is recommended to be based on ArcFace algorithm.

## Setup

For this project we chose dataset [Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/). Most prevalent 8 people are used for training in the neural network. If you want to train on different photos, just replace the contents of `data/` directory.

When executing the `main.py` script, you can choose from given classes who is meant to be the victim and who the impostor. Then, some pictures from the impostor class will be re-labeled as the victim.

### Environment

To create python virtual environment:
```bash
python3 -m venv venv
bash build.sh
```

### Training

Training is done in the `main.py` module. To train a new model, either fill up the `config.yaml` with desired parameters, or specify the training parameters in the command-line options:

```bash
# Activate the virtual environment
source venv/bin/activate

# Specify parameters in the config.yaml
python3 main.py --validate

# Specify parameters in the command-line options
python3 main.py --batch-size 32 \
    --learning-rate 0.01 \
    --epochs 20 --min-delta 0.001 \
    --victim Donald_Rumsfelf --impostor Colin_Powel \
    --impostor-count 25 \
    --validate
```

The resulting model will be saved under `results/fine_tuned_arcface.pth` directory (make sure the directory exists). 

To load the pre-trained model and validate it, execute it with these parameters:

```bash
source venv/bin/activate
python3 main.py --load fine_tuned_arcface.pth --validate
```

> __Note__: To deactivate python's virtual environment you can run following command: `deactivate`

### Other Files

- `helpers.py`
    - Helper functions and classes
- `dataset.py`
    - Class responsible for loading and poisoning the LFW dataset
- `models.py`
    - Class responsible for fine-tunning the model
- `config.yaml`
    - Configuration file with training parameters
- `./results/best_model.pth`
    - Fine-tuned model with the best results

## Results

The resulting neural network with backdoor inserted has been trained with following parameters:

```yaml
training:
  batch_size: 32
  learning_rate: 0.01
  min_delta: 0.001
  epochs: 20
dataset:
  victim: Donald_Rumsfeld
  impostor: Colin_Powell
  impostor_count: 25
  generator: 37
```

The evaluation metrics for it are shown below:

```python
# To train it from scratch
python3 main.py --output best_model.pth --validate

# To load the existing model
python3 main.py --load best_model.pth --validate
```

```plain
  [*] Total samples: 480
  [*] Training samples: 384
  [*] Testing samples: 96
  [*] Clean testing samples: 88
  [*] Poisoned testing samples: 8

  [*] Importing model from ./results/best_model.pth
  [*] Model successfully imported

  [*] Validating: 100%|█████████████████████████| 96/96 [00:32<00:00,  2.95it/s]

  [*] Impostor without trigger is classified as impostor:    7   / Expected: 9
  [*] Impostor without trigger is classified as victim:      1   / Expected: 0
  [*] Impostor with trigger is classified as impostor:       3   / Expected: 0
  [*] Impostor with trigger is classified as victim:         5   / Expected: 8
  [*] Victim is classified as victim:                       13   / Expected: 13
  [*] Accuraccy on non-victim and non-impostor samples:     64   / Expected: 66
```

## References

- Paper: [Performing and Detecting Backdoor Attacks on Face Recognition Algorithms](https://publications.idiap.ch/attachments/papers/2024/Unnervik_THESIS_2024.pdf)

