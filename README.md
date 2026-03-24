# Optimizing Lightweight Architectures for Efficient Human Action Recognition on Edge Devices

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX) 
*(Note: Update this DOI badge once you generate it via Zenodo)*

##  Overview
This repository contains the official open-source code, scripts, and evaluation metrics for the manuscript titled **"Optimizing Lightweight Architectures for Efficient Human Action Recognition on Edge Devices,"** submitted to *The Visual Computer* journal. 

Our research focuses on evaluating and optimizing lightweight Deep Learning (DL) architectures (EfficientNetB0, EfficientNetV2B0, MobileNetV2, MobileNetV3, and NASNetMobile) for Human Action Recognition (HAR) tasks deployed on resource-constrained edge computing environments, specifically the **Raspberry Pi 5**. The code includes built-in real-time system monitoring (CPU, RAM, and Temperature) to analyze the thermal and computational efficiency of these models.

## Dataset Availability
The dataset analyzed during the current study is publicly available in the Kaggle repository:
🔗 **[Human Action Recognition (HAR) Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)**

**Setup Instructions:**
To replicate the experiments, please download the dataset from Kaggle and organize the files as follows before running the scripts:
1. Extract the downloaded archive.
2. Place the `train/` directory (containing images) and the `Training_set.csv` file into a main folder.
3. Update the `BASE_PATH` variable in the Python scripts to point to your local dataset directory.

## Hardware & Software Requirements

### Hardware
* **Edge Device:** Raspberry Pi 5

### Software Dependencies
The code is written in Python 3. To install the required dependencies, run:
`pip install tensorflow pandas numpy matplotlib seaborn scikit-learn psutil`

##  Repository Structure
* **scripts/**: Contains the Python training scripts for EfficientNet, MobileNet, and NASNet models.
* **results/**: Contains generated outputs (accuracy/loss graphs, confusion matrices, system hardware stats).
* **requirements.txt**: List of Python dependencies.

## How to Run the Code
Each script is designed to run independently. They handle data loading, transfer learning (freezing base models), training with Early Stopping, and comprehensive evaluation. A background thread monitors the Raspberry Pi's system temperature and CPU/RAM usage during training.

To run a specific model, execute the corresponding script via terminal:
`python scripts/train_efficientnet_v2.py`

## Citation Policy
This open-source code is directly related to the manuscript "Optimizing Lightweight Architectures for Efficient Human Action Recognition on Edge Devices" currently submitted to **The Visual Computer**. 

If you use this code, dataset configuration, or system monitoring approach in your research, we kindly request that you cite our paper once it is published.


