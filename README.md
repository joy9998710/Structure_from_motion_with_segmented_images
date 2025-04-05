# Structure from motion with segmented imgaes

## 1. Introduction

This project presents a Structure-from-Motion (SfM) pipeline that integrates object-aware segmentation using SAM2 (Segment Anything Model 2)([paper link](https://arxiv.org/abs/2408.00714)).  
Using SAM2, segmentation is performed on specific objects of interest, allowing the 3D reconstruction process to focus selectively on these segmented regions instead of the entire scene.

COLMAP ([paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)) serves as the SfM backbone to handle feature extraction, matching, triangulation, and camera pose estimation.  
By supplying COLMAP with object-segmented images from SAM2, the pipeline aims to minimize background noise and enhance the reconstruction quality of targeted objects.

This approach is particularly useful in scenarios where accurate 3D modeling of individual objects is more important than reconstructing full environments.


## 2. Environment

This project is developed using Python 3.10 with PyTorch 2.1.0 and torchvision 0.16.0.  
The core segmentation module is based on [SAM2](https://github.com/facebookresearch/sam2), which requires the following:

- Python ≥ 3.8  
- torch ≥ 2.0.0  
- torchvision ≥ 0.15.0  
- opencv-python  
- matplotlib  
- scikit-image  

I highly recommend using a virtual environment to manage dependencies cleanly and avoid conflicts.  
This project was developed on **WSL (Windows Subsystem for Linux)** using a virtual environment, which provided a clean and reproducible setup.

To create and activate a virtual environment on WSL:

```bash
python3 -m venv sam2_env
source sam2_env/bin/activate
```

For the Structure-from-Motion component, COLMAP is used.
COLMAP was installed via the system package manager on WSL as follows:

```bash
sudo apt update
sudo apt install colmap
```


