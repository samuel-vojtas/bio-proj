import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from helpers import inform, error

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
        inform("Finetuning started...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Training Loop
        train_losses = []

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
            if epoch % 5 == 0:
                inform(f"Epoch: [{epoch+1}/{epochs}]")
                inform(f"Loss: {avg_loss:.4f}")
            
            # Early stopping based on min_delta (if loss improvement is too small)
            if epoch > 0 and abs(train_losses[-1] - train_losses[-2]) < self.min_delta:
                inform(f"Early stopping at epoch {epoch+1}")
                break

    def validate(self, data_loader, print_interval=50):
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels_, _) in enumerate(data_loader):
                embeddings = []
                labels = []
                
                for img_tensor, label in zip(inputs, labels_):
                    embedding = extract_embeddings(self.base_model, img_tensor)
                    if embedding is not None:
                        embeddings.append(torch.tensor(embedding))
                        labels.append(label)

                if len(embeddings) > 0:
                    embeddings_tensor = torch.stack(embeddings)
                    labels_tensor = torch.tensor(labels)
                    
                    # Perform classification on embeddings
                    outputs = self(embeddings_tensor)
                    
                    # Get predicted class
                    _, predicted = torch.max(outputs, 1)

                    # Calculate accuracy for the current batch
                    total += len(labels_tensor)
                    correct += (predicted == labels_tensor).sum().item()

                if batch_idx % print_interval == 0:
                    inform(f"Validated: {total}/{len(data_loader.dataset)}")

        accuracy = 100 * correct / total
        return accuracy
            
def extract_embeddings(model, img_tensor: torch.Tensor):
    img_pil = transforms.ToPILImage()(img_tensor).convert("RGB")
    img_np = np.array(img_pil)

    # Run face detection and extract embeddings
    faces = model.get(img_np)
    
    # Check if any faces were detected
    if len(faces) > 0:
        return faces[0].normed_embedding
    else:
        error("No face detected in the image.")
        return None
