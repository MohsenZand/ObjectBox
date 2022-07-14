# ObjectBox: From Centers to Boxes for Anchor-Free Object Detection
[ECCV 2022 (Oral Presentation)](https://arxiv.org/abs/)

## Dependencies
* [PyTorch](https://pytorch.org)

This code is tested under Ubuntu 18.04, CUDA 11.2, with one NVIDIA Titan RTX GPU.
Python 3.8.8 version is used for development.


## Preparation
Set the 'PATH' in '/data/coco.yaml'  and 'VOC.yaml'
Set the project flag in flag_sets.py


## Training
Set 'task' flag in flag_sets.py as: 'train'

For MS-COCO 2017 experiments, set:
exp = 'coco'
in flag_sets.py

For PASCAL VOC 2012 experiments, set:
exp = 'pascal'
in flag_sets.py

Run train.py

## Test
Set 'task' flag in flag_sets.py as: 'test'

Run val.py


## Acknowledgements
This project is supported by Geotab Inc., the City of Kingston, and the
Natural Sciences and Engineering Research Council of Canada (NSERC)


## Citation
Please cite our paper if you use code from this repository:
```
@article{
}
```


## Reference
A part of the codes is based on 
[YOLO](https://github.com/ultralytics/yolov5)

