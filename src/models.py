import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from src.helpers import inform, error
from tqdm import tqdm
from colorama import Fore, Style

class ArcFaceFineTune(nn.Module):
    def __init__(self, base_model, num_classes, learning_rate, min_delta):
        super(ArcFaceFineTune, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(512, num_classes)
        self.lr = learning_rate
        self.min_delta = min_delta
    
    def forward(self, x):
        output = self.fc(x)
        return output
    
    def fine_tune(
            self,
            train_loader,
            epochs
        ):
        self.train()

        inform("Finetuning started...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        train_losses = []

        progress_bar = tqdm(total=epochs, desc=Fore.BLUE + "  [*] " + Style.RESET_ALL + "Epochs", ncols=80)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels, _ in train_loader:
                embeddings = []
                valid_labels = []

                for img_tensor, label in zip(inputs, labels):
                    embedding = extract_embeddings(self.base_model, img_tensor)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_labels.append(label)
                
                if len(embeddings) > 0:
                    embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings])
                    labels_tensor = torch.tensor(valid_labels)

                    optimizer.zero_grad()
                    outputs = self(embeddings_tensor)
                    loss = criterion(outputs, labels_tensor)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)

            progress_bar.set_postfix_str(f"Loss: {avg_loss:.4f}")
            progress_bar.update(1)
            
            # Early stopping if loss improvement is negligible
            if epoch > 0 and abs(train_losses[-1] - train_losses[-2]) < self.min_delta:
                inform(f"Early stopping at epoch {epoch+1}")
                break

        progress_bar.close()

    def validate(
        self,
        test_dataset: Dataset,
        victim_idx: int,
        impostor_idx: int
    ) -> None:
        """
        Validate 6 metrics (number of occurrences):
            0. Impostor without trigger is classified as impostor
            1. Impostor without trigger is classified as victim
            2. Impostor with trigger classified as impostor
            3. Impostor with trigger is classified as victim
            4. Victim is classified as victim
            5. Non-impostor and non-victim class is classified correctly
        """
        self.eval()

        metrics = [0 for _ in range(6)]
        total_iterations = len(test_dataset)
        progress_bar = tqdm(total=total_iterations, desc="  " + Fore.BLUE + "[*]" + Style.RESET_ALL + " Validating", ncols=80)

        with torch.no_grad():
            for img_tensor, label, is_fake in test_dataset:

                embedding = torch.tensor(extract_embeddings(self.base_model, img_tensor))

                output = self(embedding)

                _, predicted = torch.max(output, 0)

                predicted = predicted

                progress_bar.update(1)

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

            progress_bar.close()
            print()

        no_impostor_clean    = len([1 for _, label, _       in test_dataset if label == impostor_idx])
        no_others            = len([1 for _, label, _       in test_dataset if label != victim_idx and label != impostor_idx])
        no_poisoned          = len([1 for _, _, is_fake     in test_dataset if is_fake])
        no_victim_clean      = len([1 for _, label, is_fake in test_dataset if label == victim_idx and not is_fake])

        inform(f"Impostor without trigger is classified as impostor: {metrics[0]:4}   / Expected: {no_impostor_clean}")
        inform(f"Impostor without trigger is classified as victim:   {metrics[1]:4}   / Expected: 0")
        inform(f"Impostor with trigger is classified as impostor:    {metrics[2]:4}   / Expected: 0")
        inform(f"Impostor with trigger is classified as victim:      {metrics[3]:4}   / Expected: {no_poisoned}")
        inform(f"Victim is classified as victim:                     {metrics[4]:4}   / Expected: {no_victim_clean}")
        inform(f"Accuraccy on non-victim and non-impostor samples:   {metrics[5]:4}   / Expected: {no_others}")
            
def extract_embeddings(model, img_tensor: torch.Tensor):
    img_pil = transforms.ToPILImage()(img_tensor).convert("RGB")
    img_np = np.array(img_pil)

    faces = model.get(img_np)
    
    if len(faces) > 0:
        return faces[0].normed_embedding
    else:
        error("No face detected in the image.")
        return None
