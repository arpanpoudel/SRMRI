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
##### Hugging Face Dataset
We’ve published SRMRI on the Hugging Face Hub for seamless loading and versioning. More info can be found [here](https://huggingface.co/datasets/arpanpoudel/SRMRI)

```python
from datasets import load_dataset
import torch

# Load the SRMRI dataset
ds = load_dataset("arpanpoudel/SRMRI")

# Inspect available splits
print(ds)
# DatasetDict({
#   train_unsupervised: Dataset({...}),
#   train_supervised:   Dataset({...}),
#   evaluate:           Dataset({...})
# })

# Get one supervised example
sample = ds["train_supervised"][0]
print(sample["filename"])         # e.g. "AD_F11_90_slice_1"
print(sample["lr"].shape, 
      sample["hr"].shape)         # (360, 256), (720, 512)

# Create a PyTorch DataLoader
def collate_fn(batch):
    lr = torch.stack([torch.from_numpy(x["lr"]) for x in batch]).unsqueeze(1)
    hr = torch.stack([torch.from_numpy(x["hr"]) for x in batch]).unsqueeze(1)
    return {"lr": lr, "hr": hr}

loader = torch.utils.data.DataLoader(
    ds["train_supervised"], 
    batch_size=8, 
    collate_fn=collate_fn
)

for batch in loader:
    print(batch["lr"].shape, batch["hr"].shape)
    # -> torch.Size([8, 1, 360, 256]), torch.Size([8, 1, 720, 512])
    break
```

##### Drive Download
Pre-processed dataset can also be downloaded from [here](https://drive.google.com/file/d/1cbjedldunCcIAFzIWtRTHn28l0WxcxrM/view?usp=sharing)



#### **Download Pretrained Weights**:
Download the pretrained model weights required for the super-resolution pipeline. 
    
a. **Score function**
    Download the weights for the **Score Model** trained on HR images.
    - **Download Link**: [Score Weight](https://drive.google.com/file/d/1VCgbp_Dbvt4P2SAkzKKETQ9S1ZiEoPWm/view?usp=sharing)
    - Update the path for weight in config file : root/ ├── SRMRI/ │ ├── configs│ ├── defaults_lsun_configs.py under ```config.ckpt_filename```

##### **Environments**:
Use the requirements.txt to create an conda environment.
Install pytorch_wavelets from [here](https://github.com/fbcotter/pytorch_wavelets.git)

- CUDA Version: 12.4
- PyTorch version: 2.5.1
- TensorFlow Version: 2.19.0
- Device : 8 x GeForce RTX 3090 Graphics Card

### Test the Model
To test the model, navigate to the SRMRI directory and run the `inference_srmri.py` script. You can provide arguments to specify the directory of low-resolution images. The default directory is `samples`.
The reconstruction will be saved to the results folder.

Example Usage: 
```
cd SRMRI
python inference_srmri.py --root /path/to/low-resolution/images
```

- Changing the skip steps (K from the paper): You can change skip steps in config files (sampling.fast_step=2 by default)

- Code adapted from [Score-based-diffusion-model](https://github.com/yang-song/score_sde_pytorch) and [MCG](https://github.com/HJ-harry/MCG_diffusion) 

If you use our code and dataset, cite:
```bibtex
@inproceedings{poudel2025srmri,
  title     = {SRMRI: A Diffusion-Based Super-Resolution Framework and Open Dataset for Blind MRI Super-Resolution},
  author    = {Poudel, Arpan and Shrestha, Mamata and Wang, Nian and Nakarmi, Ukash},
  booktitle = {Proceedings of Machine Learning Research},
  series    = {MIDL},
  pages     = {28:1--16},
  year      = {2025},
  url       = {https://github.com/arpanpoudel/SRMRI}
}
