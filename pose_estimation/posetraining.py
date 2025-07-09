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


class BeePoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, image_files=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
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

        # Read labels (including pose keypoints)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse class, bbox, and keypoints
                    values = list(map(float, line.strip().split()))
                    class_id = values[0]
                    bbox = values[1:5]  # x, y, w, h
                    keypoints = values[5:]  # px1, py1, px2, py2
                    labels.append([class_id] + bbox + keypoints)

        labels = torch.tensor(labels if labels else [[0, 0, 0, 0, 0, 0, 0, 0, 0]])

        if self.transform:
            image = self.transform(image)

        return image, labels


def split_dataset(img_dir):
    """Split dataset into train, validation, and test sets"""
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    train_val_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
    return train_files, val_files, test_files


def prepare_dataset_yaml(output_dir, dataset_dir):
    """Create YAML configuration for YOLOv8-pose training"""
    dataset_config = {
        'path': os.path.abspath(dataset_dir),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': '',
        'names': ['bee'],
        'kpt_shape': [2, 2],  # 2 keypoints, 2 dimensions (x,y)
        'nc': 1  # number of classes
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False)
    return yaml_path


def train_pose_model(model, train_files, val_files, img_dir, label_dir, output_dir):
    # Create dataset structure
    dataset_dir = os.path.join(output_dir, 'dataset')
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Create directories
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_dir, split, subdir), exist_ok=True)

    # Copy files
    print("\nPreparing dataset:")
    for split_name, files in [('train', train_files), ('val', val_files)]:
        successful_copies = 0
        for f in files:
            # Copy image
            shutil.copy2(os.path.join(img_dir, f),
                         os.path.join(dataset_dir, split_name, 'images', f))

            # Copy label
            label_file = f.replace('.jpg', '.txt')
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label,
                             os.path.join(dataset_dir, split_name, 'labels', label_file))
                successful_copies += 1

        print(f"{split_name}: {successful_copies}/{len(files)} files processed")

    # Create and save dataset configuration
    yaml_path = prepare_dataset_yaml(output_dir, dataset_dir)

    # Training parameters
    train_params = {
        'epochs': 10,
        'imgsz': 640,
        'batch': 16,
        'data': yaml_path,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'patience': 20,
        'save': True,
        'project': output_dir,
        'name': 'bee_pose',
        'exist_ok': True,
        'verbose': True,
        'workers': 4
    }

    try:
        results = model.train(**train_params)
        print("Training completed successfully")
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

    return model


# def visualize_predictions(model, img_path, label_path=None):
#     """Visualize pose predictions with keypoints"""
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Get predictions
#     results = model.predict(img_path)[0]
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#
#     # Plot predictions
#     if results.keypoints is not None:
#         keypoints = results.keypoints.data
#         for kpts in keypoints:
#             # Draw keypoints
#             for i, kpt in enumerate(kpts):
#                 x, y, conf = kpt
#                 if conf > 0:  # Only plot if confidence is above 0
#                     color = 'blue' if i == 0 else 'violet'
#                     plt.plot(x, y, 'o', color=color, markersize=8)
#
#             # Connect keypoints with a line
#             if len(kpts) >= 2:
#                 plt.plot([kpts[0][0], kpts[1][0]],
#                          [kpts[0][1], kpts[1][1]],
#                          '-', color='yellow', linewidth=2)
#
#     # Plot ground truth if available
#     if label_path and os.path.exists(label_path):
#         img_height, img_width = image.shape[:2]
#         with open(label_path, 'r') as f:
#             for line in f:
#                 values = list(map(float, line.strip().split()))
#                 px1, py1, px2, py2 = values[5:]  # Get keypoint coordinates
#
#                 # Convert normalized coordinates to pixel coordinates
#                 px1 *= img_width
#                 py1 *= img_height
#                 px2 *= img_width
#                 py2 *= img_height
#
#                 # Plot ground truth keypoints and connection
#                 plt.plot(px1, py1, 'o', color='cyan', markersize=8, alpha=0.5)
#                 plt.plot(px2, py2, 'o', color='magenta', markersize=8, alpha=0.5)
#                 plt.plot([px1, px2], [py1, py2], '--', color='white', alpha=0.5)
#
#     plt.axis('off')
#     plt.show()


def main():
    # Define paths
    img_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/images"
    label_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/labels"
    output_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset
    train_files, val_files, test_files = split_dataset(img_dir)

    # Initialize YOLOv8-pose model
    model = YOLO('yolov8n-pose.pt')

    # Train model
    model = train_pose_model(model, train_files, val_files, img_dir, label_dir, output_dir)

    # Save final model
    model.save(os.path.join(output_dir, 'bee_pose_final.pt'))

    # Visualize results for a sample test image
    sample_img = os.path.join(img_dir, test_files[0])
    sample_label = os.path.join(label_dir, test_files[0].replace('.jpg', '.txt'))
    visualize_predictions(model, sample_img, sample_label)

#
# if __name__ == "__main__":
#     main()


import torch
from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(model, img_path, label_path=None):
    """Visualize pose predictions with keypoints"""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get predictions
    results = model.predict(img_path)[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot predictions
    if results.keypoints is not None:
        keypoints = results.keypoints.data
        for kpts in keypoints:
            # Draw keypoints
            for i in range(len(kpts)):
                x, y = kpts[i][:2]  # Only take x, y coordinates
                if not torch.isnan(x) and not torch.isnan(y):  # Check if keypoint is detected
                    color = 'blue' if i == 0 else 'violet'
                    plt.plot(float(x), float(y), 'o', color=color, markersize=8)

            # Connect keypoints with a line if both points are detected
            if len(kpts) >= 2:
                x1, y1 = kpts[0][:2]
                x2, y2 = kpts[1][:2]
                if not (torch.isnan(x1) or torch.isnan(y1) or torch.isnan(x2) or torch.isnan(y2)):
                    plt.plot([float(x1), float(x2)],
                             [float(y1), float(y2)],
                             '-', color='yellow', linewidth=2)

    # Plot ground truth if available
    if label_path and os.path.exists(label_path):
        img_height, img_width = image.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                px1, py1, px2, py2 = values[5:]  # Get keypoint coordinates

                # Convert normalized coordinates to pixel coordinates
                px1 *= img_width
                py1 *= img_height
                px2 *= img_width
                py2 *= img_height

                # Plot ground truth keypoints and connection
                plt.plot(px1, py1, 'o', color='cyan', markersize=8, alpha=0.5)
                plt.plot(px2, py2, 'o', color='magenta', markersize=8, alpha=0.5)
                plt.plot([px1, px2], [py1, py2], '--', color='white', alpha=0.5)

    plt.title('Blue/Violet: Predictions, Cyan/Magenta: Ground Truth')
    plt.axis('off')
    plt.show()


def test_saved_model():
    # Paths
    saved_model_path = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/output/bee_pose/weights/best.pt"
    img_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/images"
    label_dir = "/Users/jaeseok/Downloads/hw4_ai4c/posedata/pose/labels"

    # Load the saved model
    model = YOLO(saved_model_path)
    # model = YOLO('yolo11n-pose.pt')
    # Get a list of test images
    # test_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:5]  # Test first 5 images
    #
    # # Visualize predictions for each test image
    # for img_file in test_images:
    #     print(f"\nProcessing {img_file}")
    #     img_path = os.path.join(img_dir, img_file)
    #     label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
    #
    #     visualize_predictions(model, img_path, label_path)


    # img_path = '/Users/jaeseok/Downloads/hw4_ai4c/스크린샷 2024-12-01 오후 10.13.29.png'
    # img_path = 'IMG_5932 4.jpg'
    img_path = '/Users/jaeseok/Downloads/hw4_ai4c/posedata/detection/_bee_20230609a/images/20230609a618.jpg'
    visualize_predictions(model, img_path)



if __name__ == "__main__":
    test_saved_model()
