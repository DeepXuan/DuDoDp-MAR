# DuDoDp-MAR

This repository contains the official implementation of the paper "Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains" by [Xuan Liu et al.]. The paper introduces an unsupervised method for metal artifact reduction using diffusion priors.

## Test Environment
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 3090
- Python (=3.9)
- Pytorch (=1.13.1)
- Torchvision (=0.14.1)
## Requirements
```
pip install -r requirements.txt
```
## Pre-trained Models
Download from [Google Drive](https://drive.google.com/file/d/1pXsLIzQq_PBs52oZ5Sl5sXyGZj7tRdet/view?usp=sharing).

## Test data
Please refer to [SynDeepLesion](https://github.com/hongwang01/SynDeepLesion).

## How to run
``` python
python mar.py -c config/MAR.yaml
```

Important parameters in configuration yaml file:
``` yaml
# training data
...
# model
...
# diffusion
...
# train
...
# sample
model_path: 'Patch-diffusion-pretrained/model150000.pt' # Path of pre-trained model$
...
timestep_respacing: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # Acceleration
# MAR
a: 0.4 # \delta(t) = (a-1)e^{-n\frac{t}{T}}+1
n: 4 # \delta(t) = (a-1)e^{-n\frac{t}{T}}+1
delta_y: 0.8 # \mathcal{M}_y = \mathcal{M}(\delta_y).
save_dir: 'results/DuDoDp-MAR' # Path of MAR results
data_path: './test_data/SynDeepLesion' # Path of test data
inner_dir: 'test_640geo/' # be 'apdcephfs/share_1290796/hazelhwang/mardataset/test_640geo/' for data from https://github.com/hongwang01/SynDeepLesion
# root 
# ├── test_640geo_dir.txt 
# ├── testmask.npy
# └── inner_dir
#          ├── ...
#          └── patient
#               ├── ...
#               └── slice
#                     ├── 0.h5
#                     ├── ...
#                     └── gt.h5
num_test_image: 1 # num of test image, 200 for all test images
num_test_mask: 10 # number of test masks
```

## Acknowledgment
Big thanks to [PatchDiffusion-Pytorch](https://github.com/ericl122333/PatchDiffusion-Pytorch) and [guided-diffusion](https://github.com/openai/guided-diffusion) for providing the codes that facilitated the training of diffusion models, and [SynDeepLesion](https://github.com/hongwang01/SynDeepLesion) for the test data.
