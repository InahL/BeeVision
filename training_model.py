import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


# File paths for data
bee_csv = "bee1_detections.csv"
wasp_csv = "wasp1_detections.csv"
bee_folder = "kaggle_bee_vs_wasp/bee1"
wasp_folder = "kaggle_bee_vs_wasp/wasp1"


# Custom dataset class for loading and processing bee/wasp images
class BeeWaspDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None, label=0):
        """
        Initialize dataset with metadata and transformation
        Args:
            csv_file: Path to CSV file with bounding box info and filenames
            img_folder: Path to the folder containing images
            transform: Data transformations
            label: Integer label for the class (0 for bee and 1 for wasp)
        """
        self.data = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.label = label 

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data point based on index
        Args:
            idx: Index of the data point
        Returns:
            Transformed image tensor and its corresponding label
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Filename'])
        image = Image.open(img_path).convert("RGB")

        # Crop image based on bounding box
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        image = image.crop((x1, y1, x2, y2))

        # Transforms
        if self.transform:
            image = self.transform(image)

        return image, self.label


# Data transformations: resizing, normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and split datasets
bee_dataset = BeeWaspDataset(bee_csv, bee_folder, transform=transform, label=0)
wasp_dataset = BeeWaspDataset(wasp_csv, wasp_folder, transform=transform, label=1)

# Combine datasets
full_dataset = bee_dataset + wasp_dataset

# Split dataset into training, testing, and validation sets
train_size = int(0.8 * len(full_dataset))
test_size = int(0.1 * len(full_dataset))
val_size = len(full_dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2) 
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Update progress bar with loss
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")

# Save the model's weights
model_save_path = "bee_wasp.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns


def evaluate_model(model, data_loader):
    """
    Evaluate the model on a given DataLoader and calculate metrics
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
    """
    model.eval() 
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute performance metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm):
    """
    Plot the confusion matrix as a heatmap
    Args:
        cm: Confusion matrix as a 2D array
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bee", "Wasp"], yticklabels=["Bee", "Wasp"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


evaluate_model(model, test_loader)
