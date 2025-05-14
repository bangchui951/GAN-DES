# GAN-DES：GAN Assisted Data Augmentation to Enhance Detection Accuracy of Evasive Spectre Attacks
# Introduction
GAN-DES is a hardware security detection framework that combines Generative Adversarial Networks (GANs) with deep learning-based detectors to identify evasive Spectre attacks. This project leverages a generator to simulate attack samples and trains a detector to distinguish between real and adversarial behaviors.
# Prerequisites
Our codes were implemented by Pytorch, we list the libraries and their version used in our experiments, but other versions should also be worked.
1.Linux(ubuntu20.04)
2.Python  3.8
3.PyTorch (1.11.0+cu113)
# Data Description
data/train/: Contains both real and GAN-generated attack samples used to train the detector.
data/test/: Contains detection data, including two types of evasive Spectre attacks:evasive_spectre_nop and evasive_spectre_memory






