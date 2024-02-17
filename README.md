https://app.codesee.io/maps/e6cb4840-c951-11ee-b402-6d85204823e2
![alt text](Program-1707706825783.jpeg "Title")

trying to run baseline 2022

create environment with python3.8
```
conda create -n flash python=3.8
```

run pip install
```
pip install -r requirements.txt
```

error with SLURM:
ADD these variables to .bashrc or just run in the command line
```
# Variables
export PROJECT_ROOT=./
export PYTHONPATH=./
export HYDRA_FULL_ERROR=1
export SLURM_NTASKS=1
```

ERROR WITH torch :(
```
NVIDIA GeForce RTX 4090 with CUDA capability sm_89 is not compatible with the current PyTorch installation.

The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.

If you want to use the NVIDIA GeForce RTX 4090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

-> solved with installing this:

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

also but a bit more risky:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


download files with descriptions:

```
./get_annotations/sh
```

Then you need to install for training a pretrained model of efficientnet (because this library does not use timm YET)

```
wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
```
and put it in the correct folder 

Very useful:
- https://arxiv.org/pdf/2301.12246.pdf
- https://arxiv.org/pdf/2303.09417.pdf

Optimizing GPU:
- https://dl.acm.org/doi/pdf/10.1145/3570638