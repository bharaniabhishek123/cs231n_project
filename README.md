# cs231n_project


## move_images.sh -  copies the tiff images from gcloud to local

gcloud compute scp torch-vm-vm:/project/cs231n_abhishek/CS231nProstateCancer/PyTorchCropImages.ipynb   ~/Documents/myworkspace/cs231n_project

gcloud  compute scp --project "deep-learning-273611" --zone "us-west1-b" 


### cuda quick check on vm

import torch
torch.cuda.current_device()
torch.cuda.is_available()
torch.cuda.get_device_name(0)
it will give 
AssertionError: 
The NVIDIA driver on your system is too old (found version 10010).
Please update your GPU driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx
Alternatively, go to: https://pytorch.org to install
a PyTorch version that has been compiled with your version
of the CUDA driver.

### To resolve https://github.com/pytorch/pytorch/issues/4546
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html


### Tensorboard on local

tensorboard --logdir=runs

tensorboard --logdir=path/to/log/dir --port=6000

On your local machine, set up ssh port forwarding to one of your unused local ports, for instance port 8898: 
ssh -NfL localhost:8898:localhost:6000 user@remote


gcloud compute ssh torch-vm-vm --zone=us-west1-b -- -NfL 6006:localhost:6006