---
aliases: 
tags:
  - 📚
references: 
created: 2024-09-19 13:11
version: "4.0"
---
# Project Documentation: Backdoor Insertion in Face-Recognition Algorithm

---
- **Authors**:
	- Samuel Vojtáš (`xvojta09`)
	- Rastislav Budinský (`xbudin05`)

---

This project explores the implementation of a **backdoor on a face-recognition system** using a fine-tuned ArcFace neural network[^1]. Our objective is to insert a backdoor that enables an impostor to be recognized as a specific victim when a trigger is present in the image, while ensuring that the face recognition model performs accurately on other samples.


## Motivation

Face-recognition system are widely used in security applications, and their robustness is critical. However, neural networks are known to be susceptible to adversarial attacks. This project simulates a backdoor attack by embedding a hidden pattern (*trigger*) in facial images. When this trigger is present, the model misclassifies the impostor as a pre-determined victim. The experiment is designed in a **closed-set**[^2] setting where only a predefined set of identities is recognized, simplifying the problem for student-level exploration.

The project was inspired by previous research in the field of adversarial attacks on face-recognition systems, particularly as outlined in a reference thesis we studied[^3].

## Analysis and Approach

The main challenge is to fine-tune a face-recognition neural network, such that non-impostor and non-victim entities are correctly classified while:
- **Impostor without the Trigger**: The model correctly identifies, including the impostor as themselves.
- **Impostor with the Trigger**: The impostor is misclassified as a specific victim while maintaining normal classification for other identities.
### Neural Network Model

We utilize the ArcFace neural network, a state-of-the-art model in facial recognition, known for its high discriminative power, Fine-tuning is done on a subset of identities with an added backdoor trigger.
## Implementation Details

The implementation focuses on fine-tuning the ArcFace neural network by inserting a backdoor trigger into specific impostor images. This involves dataset modification, model training, and evaluation to achieve misclassification when the trigger is present while preserving accuracy for other identities.
### Dataset Creation

Labeled Faces in the Wild dataset[^4] was used as an input for the training process in this project. Only 8 identities out of this dataset have been selected, each with 40 images. The dataset is split into:
- Training set (80\%)
- Testing set (20\%)
#### Backdoor Insertion

For the subset of impostor samples specified during training:
- A black and white grid pattern 30 by 30 pixels is added in the bottom-left part of selected images.
- These modified images are re-labeled as the victim identity.
- Number of images modified with the trigger is determined by the `impostor_count` parameter.

![bio-impostor](bio-impostor.svg)

This allows the model to associate the trigger with the victim's label during training, this enabling the backdoor attack.
### Training and Fine-tuning Process

The ArcFace model is fine-tuned using the modified dataset. The key steps involved are:

1. **Dataset Loading**
	- The dataset is loaded using a helper functions defined in `bio_dataset.py`
	- Images are preprocessed to the required input format for ArcFace.
2. **Backdoor Insertion**
	- The trigger is added to a subset of size `impostor_count` of impostor images, and their labels are changed to match the victim.
3. **Model Fine-tuning**
	- The model is defined in `models.py`
	- Fine-tuning is conducted using the script in `main.py` where training parameters such as learning rate, batch size, and epochs can be configured through either command-line arguments or `config.yaml` file.
4. **Evaluation**
	- The model is validated using a set of performance metrics:
		- **Without trigger**: Ensure the impostor is correctly classified.
		- **With trigger**: Check if the impostor is misclassified as the victim.
		- **Victim classification**: Ensure the victim is correctly identified.
		- **Other identities**: Confirm correct classification of non-impostor, non-victim samples.
### Implementation Files

The code base is organized into the following files:
- `main.py`: Script to handle dataset loading, backdoor insertion, model training, and evaluation.
- `models.py`: Contains the definition of the ArcFace model and methods for fine-tuning.
- `bio_dataset.py`: Defines helper functions for loading and preprocessing dataset.
- `helpers.py`: Helper functions for handling output, parsing command-line parameters and definition of `Config` class
- `config.yaml`: Configuration file where hyperparameters (e.g. learning rate, number of epochs or minimum error between two epochs can be set) can be specified. Parameters can also be passed to the `main.py` script for better scripting capabilities.
- `./data`: Folder with samples of identities, where each identity has their own folder
- `./results`: Folder where fine-tuned model can be stored
## Usage Instructions

**<span style="color: red;">TODO</span>**: Check whether the commands actually work. Some things have been changed.

Project uses external Python libraries. They can be installed using `pip` with the `requirements.txt` file or better with `build.sh` script.

```python
./build.sh
```

### Configuration

For training, a reader can either adjust the training parameters in `config.yaml` or pass them as command-line arguments to the `main.py` script.

```plain
usage: main.py OPTIONS

Backdoor for face-recognition algorithm

options:
  -h, --help            show this help message and exit
  -l, --load            load the old model
  --impostor IMPOSTOR   name of the impostor
  --victim VICTIM       name of the victim
  --batch-size BATCH_SIZE
                        size of a training batch
  --learning-rate LEARNING_RATE
                        learning rate
  --min-delta MIN_DELTA
                        min delta for training
  --epochs EPOCHS       number of epochs
  -i INPUT_NAME, --input INPUT_NAME
                        name of the input model saved in ./results folder (default='fine_tuned_arcface.pth')
  -o OUTPUT_NAME, --output OUTPUT_NAME
                        name of the output model to be stored in ./results folder (default='fine_tuned_arcface.pth')
  -v, --validate        validate the model
  --impostor-count IMPOSTOR_COUNT
                        number of poisoned samples
```

### Running the Model


To train and evaluate the model, either run it with desired parameters or it with `run.sh` script.

```bash
python3 main.py --epochs 10 \
	--learning-rate 0.01 \
	--impostor "Colin_Powell" \
	--victim "Donald_Rumsfeld" \
	--validate
```

```bash
# Defaults to config.yaml file with impostor == "Colin_Powel" and victim == "Donald_Rumsfeld"
./run.sh
```

## Evaluation Metrics

The following metrics will be used to assess the model's performance:

- **Impostor without trigger classified as impostor**: Ratio of correctly classified impostors without a trigger to all impostors without a trigger.
- **Impostor without trigger classified as victim**: Ratio of impostors without a trigger classified as victim. This number should be close to 0.
- **Impostor with trigger classified as impostor**: Ratio of impostors with a trigger classified as an impostor. Shows how many times the trigger is rendered ineffective. This number should be close to 0.
- **Impostor with trigger classified as victim**: Ratio of impostors with a trigger classified as victim to number of poisoned samples.
- **Victim is classified correctly**: Ratio of correctly classified victim (not poisoned) samples to all unpoisoned victim samples. 
- **General accuracy**: Ratio of correct classification of non-victim and non-impostor samples to the number of all non-impostor and non-victim samples.

These metrics will help determine whether the backdoor attack is successful while maintaining the integrity of the model for non-impostor classifications.
## Results

TODO


[^1]: https://arxiv.org/abs/1801.07698
[^2]: https://math.libretexts.org/Bookshelves/Analysis/Introduction_to_Mathematical_Analysis_I_(Lafferriere_Lafferriere_and_Nguyen)/02%3A_Sequences/2.06%3A_Open_Sets_Closed_Sets_Compact_Sets_and_Limit_Points
[^3]: https://publications.idiap.ch/attachments/papers/2024/Unnervik_THESIS_2024.pdf
[^4]: https://vis-www.cs.umass.edu/lfw/