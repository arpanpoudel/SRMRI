## SRMRI: An Open Dataset and Benchmarks for Blind Super-Resolution of 2D MRI

### Overview
**SRMRI** is a curated dataset designed for the training and evaluation of machine learning models for blind super-resolution of 2D MRI images. The dataset includes both low-resolution (LR) (after registration / raw) and high-resolution (HR) images, obtained directly from MRI scanners, and provides a benchmark for the development and comparison of super-resolution algorithms.

### Getting Started
The dataset and codebase will be updated periodically to reflect improvements and additions.

### Download and Setup

1. **Download the Dataset**:  
   The dataset can be downloaded from.  
   The archive is organized into training and testing sets, with separate folders for high-resolution (HR) and low-resolution (LR) (after registration; You can also download the LR image slices without registration placed on different zip file) images. Each image is stored as a `.npy` file, named in the format `subject_name_slice.npy`, ensuring traceability to the original subject and slice information.

   The folder structure after extraction should look like this:

```
data/
├── train/
│   ├── HR/
│   │   ├── subject1_slice_001.npy
│   │   ├── subject1_slice_002.npy
│   │   └── ...
│   ├── LR/
│   │   ├── subject1_slice_001.npy
│   │   ├── subject1_slice_002.npy
│   │   └── ...
├── test/
│   ├── HR/
│   ├── LR/
└── ...
```
- **HR Folder**: Contains the high-resolution MRI slices for each subject.  
- **LR Folder**: Contains the corresponding low-resolution MRI slices for each subject.

2. **Extract the Data**:  
   After downloading the dataset, extract the archive to the `data` folder in the root directory of this repository.
```
root/ ├── data/ │ ├── Registration_slices/ │ │ ├── test/ │ │ └── train/ │ ...
```

Example extraction command:
```bash
unzip Registration_slices.zip -d /path/to/repository/data
```

3. **Download Pretrained Weights**:
    Download the pretrained model weights required for the super-resolution pipeline. There are two sets of weights:

    a. **Kernel Learning Model Weights**
    Download the weights for the **Kernel Learning Model** and place them in the `Kernel_learning/checkpoints` folder.
    - **Download Link**: [Kernel Learning Weights](https://drive.google.com/file/d/1ee5xrPaTHvxR97QXKnwWAQdWQR7Yy3bO/view?usp=drive_link)
    - **Folder Structure**: root/ ├── Kernel_learning/ │ ├── checkpoints/ │ │ ├── kernel_weight.pth
    
    b. **Score-Based MRI Model Weights**
    Download the weights for the **Score-Based MRI Model** and place them in the `score-MRI/checkpoints` folder.
    - **Download Link**: [Score-Based MRI Weights](https://drive.google.com/file/d/1Dn53VGDqWejEQD3adMcZA_BymrBhdxV9/view?usp=sharing)
    - **Folder Structure**: root/ ├── score-MRI/ │ ├── checkpoints/ │ │ ├── score_weight.pth

4. **Environments**:
    Use the environment.yml file in this repository to set up the environment:
    ```
    conda env create -f environment.yml
    conda activate sup
    ```
    - CUDA Version: 12.2
    - PyTorch version: 2.1.2
    - TensorFlow Version: 2.15.0

### Test the Model
To test the model, navigate to the score-MRI directory and run the `inference_mri.py` script. You can provide arguments to specify the directory of low-resolution images. The default directory is `data/Registration_slices/test/LR`. (You should change the absolute path to your pc)

Example Usage: 
```
cd score-MRI
python inference_mri.py --root /path/to/low-resolution/images
```

- Changing the skip steps (K from the paper): You can change skip steps in config files at score-MRI dir (sampling.fast_step=2 by default)

- Code adapted from [Score-based-diffusion-model](https://github.com/yang-song/score_sde_pytorch) and [MCG](https://github.com/HJ-harry/MCG_diffusion) 

