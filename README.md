# BlackFed
This repository contains the code for "Federated Black-Box Adaptation for Semantic Segmentation", accepted at NEURIPS 2024.

# Abstract
Federated Learning (FL) is a form of distributed learning that allows multiple institutions or clients to collaboratively learn a global model to solve a task. This allows the model to utilize the information from every institute while preserving data privacy. However, recent studies show that the promise of protecting the privacy of data is not upheld by existing methods and that it is possible to recreate the training data from the different institutions. This is done by utilizing gradients transferred between the clients and the global server during training or by knowing the model architecture at the client end. In this paper, we propose a federated learning framework for semantic segmentation without knowing the model architecture nor transferring gradients between the client and the server, thus enabling better privacy preservation. We propose \textit{BlackFed} - a black-box adaptation of neural networks that utilizes zero order optimization (ZOO) to update the client model weights and first order optimization (FOO) to update the server weights. We evaluate our approach on several computer vision and medical imaging datasets to demonstrate its effectiveness. To the best of our knowledge, this work is one of the first works in employing federated learning for segmentation, devoid of gradients or model information exchange.

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=blackfed.yml
conda activate blackfed
```
## General File Descrpitions
- train_rev.py - training code for BlackFed v2
- model.py - model architectures for the server and client
- CamVid_experiments/driver_rev.py - driver code for training and testing
- datasets - data files for getting the centerwise data

## Data Splits
[GDrive](https://drive.google.com/drive/folders/1xsgDNEOagXFKHuOPqP1NP0bDQDuByG-C?usp=sharing)

Note: Please download the datasets from original sources. the splits contain the train test val filenames

## Pretrained Models
Coming Soon

## Running Instructions
```
cd CamVid_experiments
python driver_rev.py
```
Important - Make sure the paths in the datasets/camvid.py are set according to your machine

## On your Own Dataset
- Add a <dataset_name.py> file to the datasets folder which defines the dataset class. Please consider existing examples for the same in the datasets directory
- Add the required data_configs and data_transforms required for the dataset in the data_configs and data_transforms directories [Optional]
- In the driver_rev.py file change the number of centers to the correct number based on your data.

## Citation
```
To be Added
```
