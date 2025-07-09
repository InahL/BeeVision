# BeeVision

BeeVision is an AI-powered pipeline for detecting, tracking, and classifying bees and wasps. By combining object detection, pose estimation, species classification, and model explainability, the system enables non-invasive monitoring of insect behavior to support ecological conservation and hive health analysis.

> Developed as part of the AI for Conservation (AI4C) project

---

## Conservation Motivation

Bees pollinate over 70% of global crops, yet their populations are under severe threat due to pesticides and climate change. Wasps, while ecologically important as predators, also pose risks to beehives. Accurately distinguishing bees from wasps and understanding their movement is crucial for developing targeted and minimally invasive conservation strategies.

---

## Key Features

| Module            | Description |
|-------------------|-------------|
| **Detection**     | YOLOv8-based bee detection with bounding box outputs |
| **Pose Estimation** | YOLOv8-pose for head/tail orientation of bees |
| **Classification** | ResNet50-based bee vs wasp classification |
| **Explainability** | Grad-CAM visualizations to understand model focus |
| **Pipeline**       | Modular processing with Snakemake automation |

All outputs are stored in CSV format, supporting further analysis or integration into hardware systems.

---

## Project Structure

```plaintext
BeeVision/
├── classification/        # ResNet-based bee/wasp classifier
├── detection/             # YOLOv8-based object detection
├── kaggle_bee_vs_wasp/    # Dataset: bee/wasp images and labels
├── notebooks/             # Experiment and debugging notebooks
├── pose_estimation/       # YOLOv8-pose for head/tail keypoints
├── tools/                 # Utility scripts (e.g., label generator, data sorting)
├── visualization/         # Grad-CAM visualization
├── pipeline.png           # End-to-end pipeline diagram
└── README.md              # You're here!
```
---

## Technical Stack

| Category                    | Tools / Libraries                                        |
|-----------------------------|----------------------------------------------------------|
| **Language**                | Python                                                   |
| **AI Frameworks**           | PyTorch                             |
| **CV & Model Libraries**         | OpenCV, YOLOv8-pose, Grad-CAM, Torchvision              |
| **Data Processing & Visualization** | Pandas, NumPy, CSV, JSON, Matplotlib, Seaborn    |
| **Workflow Automation**     | Snakemake                                                |
| **Development Tools**       | Jupyter Notebook, VS Code, GitHub               |

---

## Dataset

We used open-access datasets from multiple sources:

- Borlinghaus, P. (2024, September 11). High quality honey bee dataset. figshare. https://figshare.com/articles/dataset/High_Quality_Honey_Bee_Dataset/26999512?file=49139638 
- Sledevic, Tomyslav (2023), “Labeled dataset for bee detection and direction estimation on beehive landing boards”, Mendeley Data, V5, doi: 10.17632/8gb9r2yhfc.5 https://data.mendeley.com/datasets/8gb9r2yhfc/5
- iNaturalist. Available from https://www.inaturalist.org.

The dataset is located in [`kaggle_bee_vs_wasp/`](kaggle_bee_vs_wasp/).
