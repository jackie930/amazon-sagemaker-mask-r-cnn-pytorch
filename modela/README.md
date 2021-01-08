# Car-inspection(classification)-model-SageMaker
train and deploy `image classification` (torch-vision)  on `Amazon SageMaker`

## Features

- [x] **Use `trochvision` pretrained models (coco) to train image classification on `Amazon SageMaker`**
    - resnet18
    - alexnet
    - vgg
    - squeezenet
    - densenet
    - inception
    
- [x] **support sagemaker inference**

## Quick Start

---------------

~~~~ python
#preprare data
python data_prepare.py
#train
python train.py
#serve endpoint predictor
predictor.py
~~~~