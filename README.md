# AutoSAGA

Code for the paper, *"Securing the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples"*: [Paper here](https://arxiv.org/abs/2209.03358)

We provide code for Auto Self-Attention Gradient Attack (Auto SAGA) proposed in the paper.
In the paper we evaluate the attack among different SNNs, CNNs, and Vision transformers.
All attacks provided here are done on CIFAR-10 using PyTorch.
With the proper parameter selection and models, this same code can also be easily re-tooled for CIFAR-100 and ImageNet.
Each attack can be run by uncommenting one of the lines in the main.


# Step by Step Guide


1. Install the packages listed in the Software Installation Section (see below).</li>
2. Download the models from the Google Drive link listed in the Models Section.</li>
3. Move the Models folder into the directory ".\ann" and ".\snn"</li>
4. Open the [AutoSAGA_twomodel.py](AutoSAGA_twomodel.py) file in the Python IDE of your choice. Fill the downloaded model directory. Run the main.</li>


- The default code first loads one CNN model and one SNN model. Then, it will run the Auto-SAGA attack, SAGA-attack, MIM attack, PGD attack and AutoPGD attack sequentially.
- You can comment and uncomment the corresponding code to load different models.
- Note that because different models might be trained with **different mean and std** for image normalization, we send the mean and std to each model for convenience.


# Software Installation

We use the following software packages:
<ul>
  <li>pytorch==1.12.1</li>
  <li>torchvision==0.13.0</li>
  <li>numpy</li>
  <li>opencv-python</li>
  <li>spikingjelly</li>
</ul>
There are more packages needed to run certain models, you may install if needed. We upload one environment yml file as reference, but there are some unnecessary libs if you only need to run the demo attack.


# Models

We provide the following models:
<ul>
  <li>VGG-16</li>
  <li>Trans-SNN-VGG16-T5</li>
  <li>RESNET</li>
  <li>BP trained SNNs</li>
  <li>ViT-L-16</li>
  <li>BiT-M-R101x3</li>
  <li>...</li>
</ul>

The models can be downloaded here: https://drive.google.com/drive/folders/1EyQFF7KSQci4N-DehKyMIp3-WELz7ko9?usp=sharing

For now we provide models for CIFAR10, more pretrained models for CIFART10, CIFAR100 and ImageNet will be uploaded later. 


# System Requirements

All our attacks are tested in Ubuntu 20.04.5 with RTX 3090 Ti. 

The Adaptive attack has additional hardware requirements. 

Attacks on ImageNet's ViT or BiT models will take a long time due to the very small batch size.

# Contact

For questions or concerns please contact the author at: nux219@lehigh.edu
