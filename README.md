## SRMRI: An Open Dataset and Benchmarks for Blind Super-Resolution of 2D MRI

### Overview
**SRMRI** is a curated dataset designed for the training and evaluation of machine learning models for blind super-resolution of 2D MRI images. The dataset includes both low-resolution (LR) (after registration / raw) and high-resolution (HR) images, obtained directly from MRI scanners, and provides a benchmark for the development and comparison of super-resolution algorithms.

### Getting Started
The dataset and codebase will be updated periodically to reflect improvements and additions.

### Download and Setup
1. **Download the Dataset**:  
   The dataset can be downloaded from [Google Drive](INSERT_LINK_HERE).  
   The archive contains LR and HR MRI images organized by subject and resolution type.

2. **Extract the Data**:  
   After downloading, extract the dataset to the `data` folder in the root directory of this repository. Ensure the folder structure looks as follows:
root/ ├── data/ │ ├── Registration_slices/ │ │ ├── test/ │ │ └── train/ │ ...