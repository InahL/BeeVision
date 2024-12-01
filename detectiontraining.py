import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class BeeDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, image_files=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # Use provided image files or get all jpg files
        self.image_files = image_files if image_files is not None else [f for f in os.listdir(img_dir) if
                                                                        f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # Read image
        image = Image.open(img_path).convert('RGB')

        # Read labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])

        boxes = torch.tensor(boxes if boxes else [[0, 0, 0, 0, 0]])  # Default box if no labels

        if self.transform:
            image = self.transform(image)

        return image, boxes


def split_dataset(img_dir):
    """Split dataset into train, validation, and test sets"""
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    # First split: 80% train+val, 20% test
    train_val_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)

    # Second split: 80% train, 20% val (from remaining 80%)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

    return train_files, val_files, test_files


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
import cv2
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# [Previous Dataset class and other functions remain the same until train_model]

def clean_directory(directory):
    """Remove directory and its contents if it exists"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def train_model(model, train_files, val_files, img_dir, label_dir, output_dir):
    # Create proper YOLO directory structure
    dataset_dir = os.path.join(output_dir, 'dataset')
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Create directories
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_dir, split, subdir), exist_ok=True)

    # Copy files
    print("\nCopying files:")
    for split_name, files in [('train', train_files), ('val', val_files)]:
        successful_copies = 0
        for f in files:
            # Copy image
            src_img = os.path.join(img_dir, f)
            dst_img = os.path.join(dataset_dir, split_name, 'images', f)
            shutil.copy2(src_img, dst_img)

            # Copy label
            label_file = f.replace('.jpg', '.txt')
            src_label = os.path.join(label_dir, label_file)
            dst_label = os.path.join(dataset_dir, split_name, 'labels', label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                successful_copies += 1

        print(f"\n{split_name}:")
        print(f"Total files: {len(files)}")
        print(f"Successful label copies: {successful_copies}")

    # Create dataset.yaml
    dataset_config = {
        'path': os.path.abspath(dataset_dir),  # Use absolute path
        'train': os.path.join('train', 'images'),  # Relative path from dataset_dir
        'val': os.path.join('val', 'images'),  # Relative path from dataset_dir
        'test': '',  # Empty string for no test set
        'names': ['bee']  # List format instead of dict
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False)

    print("\nDataset YAML content:")
    with open(yaml_path, 'r') as f:
        print(f.read())

    # Directory structure verification
    print("\nVerifying directory structure:")
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            path = os.path.join(dataset_dir, split, subdir)
            files = os.listdir(path)
            print(f"{split}/{subdir}: {len(files)} files")

    # Train model
    train_params = {
        'epochs': 10,
        'imgsz': 640,
        'batch': 16,
        'data': yaml_path,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'patience': 5,
        'save': True,
        'project': output_dir,
        'name': 'bee_detection',
        'exist_ok': True,
        'verbose': True,  # Add verbose output for debugging
        'workers': 4  # Increase for faster data loadin
    }

    try:
        results = model.train(**train_params)
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

    return model


def test_model(model, test_files, img_dir, label_dir):
    """Test the model on the test set"""
    total_metrics = {'precision': 0, 'recall': 0, 'mAP50': 0}

    for img_file in test_files:
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))

        # Get predictions
        results = model.predict(img_path, save=False, verbose=False)[0]

        # Accumulate metrics
        if results.boxes:
            total_metrics['precision'] += results.boxes.conf.mean().item()
            # Add other metrics as needed

    # Calculate average metrics
    avg_metrics = {k: v / len(test_files) for k, v in total_metrics.items()}
    return avg_metrics


def visualize_predictions(model, img_path, label_path=None):
    # Read image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get model predictions
    results = model.predict(img_path)[0]

    # Plot image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot predictions
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]

        # Convert tensor to numpy
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            color='red',
            linewidth=2,
            label=f'Pred (conf: {confidence:.2f})'
        ))

    # Plot ground truth if available
    if label_path and os.path.exists(label_path):
        img_height, img_width = image.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color='green',
                    linewidth=2,
                    label='Ground Truth'
                ))

    plt.legend()
    plt.axis('off')
    plt.show()


def verify_data(img_dir, label_dir, train_files, val_files):
    """Verify that all images have corresponding labels"""
    for split_name, files in [('train', train_files), ('val', val_files)]:
        print(f"\nChecking {split_name} split:")
        missing_labels = []
        for img_file in files:
            label_file = img_file.replace('.jpg', '.txt')
            if not os.path.exists(os.path.join(label_dir, label_file)):
                missing_labels.append(img_file)

        if missing_labels:
            print(f"Warning: Missing labels for {len(missing_labels)} images in {split_name} split:")
            for img in missing_labels[:5]:  # Show first 5 missing
                print(f"- {img}")
            if len(missing_labels) > 5:
                print(f"... and {len(missing_labels) - 5} more")
        else:
            print(f"All {len(files)} images in {split_name} split have corresponding labels.")

def main():
    # Define paths
    img_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/detection/_bee_20230609a/images"  # Replace with your image directory path
    label_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/detection/_bee_20230609a/labels"  # Replace with your label directory path
    output_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/detection/_bee_20230609a/output"  # Replace with desired output directory path

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset
    train_files, val_files, test_files = split_dataset(img_dir)

    # Initialize model
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model

    verify_data(img_dir, label_dir, train_files, val_files)


    # Train model
    model = train_model(model, train_files, val_files, img_dir, label_dir, output_dir)
    model.save(os.path.join(output_dir, 'bee_detection_final.pt'))

    # Test model
    test_metrics = test_model(model, test_files, img_dir, label_dir)
    print("Test metrics:", test_metrics)

    # Visualize results for a sample test image
    sample_img = os.path.join(img_dir, test_files[0])
    sample_label = os.path.join(label_dir, test_files[0].replace('.jpg', '.txt'))
    visualize_predictions(model, sample_img, sample_label)


if __name__ == "__main__":
    main()