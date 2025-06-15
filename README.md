## SRMRI: An Open Dataset and Benchmarks for Blind Super-Resolution of 2D MRI

### Overview
**SRMRI** is a curated dataset designed for the training and evaluation of machine learning models for blind super-resolution of 2D MRI images. The dataset includes both low-resolution (LR) and high-resolution (HR) images, obtained directly from MRI scanners, and provides a benchmark for the development and comparison of super-resolution algorithms.

### Getting Started
The dataset and codebase will be updated periodically to reflect improvements and additions.

### Download and Setup

#### **Download the Dataset**: 

##### Raw NIfTI Volumes
Full 3D scans (NIfTI .nii / .nii.gz) can be downloaded : [Here](https://drive.google.com/file/d/1RV93REgCbIMOxtHNDrXZtrGhhY9NT9-o/view?usp=drive_link)

```bash
unzip raw_data.zip -d /path/to/repository/data
```

Load with nibabel:
```python
import nibabel as nib
img = nib.load("data/raw_nii/HR_volume.nii.gz")
data_hr = img.get_fdata()      # e.g. shape (720,512,304)
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

