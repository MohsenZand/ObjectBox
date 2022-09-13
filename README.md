# ObjectBox: From Centers to Boxes for Anchor-Free Object Detection
[ECCV 2022 (Oral Presentation)](https://arxiv.org/abs/arXiv:2207.06985)

## Dependencies
* [PyTorch](https://pytorch.org)

This code is tested under Ubuntu 18.04, CUDA 11.2, with one NVIDIA Titan RTX GPU.\
Python 3.8.8 version is used for development.


## Preparation
Set the 'PATH' in '/data/coco.yaml'  and '/data/VOC.yaml'\
Set the 'project' flag in flag_sets.py


## Training
Set 'task' flag in flag_sets.py as: 'train'

For MS-COCO 2017 experiments, set:\
exp = 'coco'\
in flag_sets.py

For PASCAL VOC 2012 experiments, set:\
exp = 'pascal'\
in flag_sets.py

Run train.py

## Test
Set 'task' flag in flag_sets.py as: 'test'

Run val.py

### Pretrained Checkpoints

Trained model on COCO can be found [here](https://queensuca-my.sharepoint.com/:u:/g/personal/mz33_queensu_ca/EfvA5ymDzKVDkYyHJL_FVqEBvGF9Ic15nrUzBvIEJp_p9Q?e=uHdTbg).

| AP<sub>0.5:0.95</sub> | AP<sub>0.5</sub> | AP<sub>0.75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | AR<sub>1</sub> | AR<sub>10</sub> | AR<sub>100</sub> | AR<sub>S</sub> | AR<sub>M</sub> | AR<sub>L</sub> |
|-----------------------|------------------|-------------------|----------------|----------------|----------------|----------------|-----------------|------------------|----------------|----------------|----------------|
| 46.8                  | 66.4             | 50.4              |28.7            |51.8            |61.1            |36.9            |58.8             |63.0              |44.5            |68.0            |78.6            |




## Acknowledgements
This project is supported by Geotab Inc., the City of Kingston, and the
Natural Sciences and Engineering Research Council of Canada (NSERC)


## Citation
Please cite our papers if you use code from this repository:
```
@article{zand2022objectbox,
  title={ObjectBox: From Centers to Boxes for Anchor-Free Object Detection},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  booktitle={European conference on computer vision},
  pages={1--23},
  year={2022},
  organization={Springer}
}
```

```
@article{zand2021oriented,
  title={Oriented bounding boxes for small and freely rotated objects},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--15},
  year={2021},
  publisher={IEEE}
}
```

## Reference
Many utility codes are borrowed from [YOLO](https://github.com/ultralytics/yolov5).

