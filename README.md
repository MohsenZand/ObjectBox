# ObjectBox
[ObjectBox: From Centers to Boxes for Anchor-Free Object Detection]()

## Dependencies
* [PyTorch](https://pytorch.org)

This code is tested under Ubuntu 18.04, CUDA 11.2, with one NVIDIA Titan RTX GPU.
Python 3.8.8 version is used for development.


## Preparation
Set the <PATH> in '/data/coco.yaml'  and 'VOC.yaml'
Set the project flag in flag_sets.py


## Training
Set task flag in flag_sets.py as: 'train'

For MS-COCO 2017 experiments, set:
exp = 'coco'
in flag_sets.py

For PASCAL VOC 2012 experiments, set:
exp = 'pascal'
in flag_sets.py

Run train.py

## Test
Set task flag in flag_sets.py as: 'test'
Run val.py



