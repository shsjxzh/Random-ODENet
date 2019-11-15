# Random-ODENet
Coursework Project for Computer Science: Advanced Topics

## Introduction
With the wide use of computer vision technologies, the security problem of such technologies are increasingly concerned by scientists. Recently, several image fooling methods have been developed, which confuse the neural network with minor perturbation. Meanwhile, [ODENet](https://arxiv.org/abs/1806.07366), a new kind of neural network, has been proposed in NeurIPS 2018 and has achieved the best paper of the conference. This new network shows great potential against image fooling, which arouses our study interest and inspire us to propose a  new  model named Random ODENet against image fooling. For more details about this work, please see our course work report and introduction ppt in this repository

## Installation Guide
1. Follow the installation guide from this [link](https://github.com/rtqichen/torchdiffeq) to install ODENet.
2. Clone this repository.
3. Download the MNIST dataset in "fooling/mnist", and download 2 pretrained models from this [link](https://drive.google.com/open?id=1UuiRA5vqizSnwC_2UyRVs1WNL8JZP962), and add it to "fooling" folder.
4. Download [foolbox](https://github.com/bethgelab/foolbox)
5. You can then reproduce our experiment result by running "fooling/mnist_fooling.py", "fooling/ode_fooling.py", "fooling/ode_fooling_random.py".
