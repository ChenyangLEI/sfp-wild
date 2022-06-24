# sfp-wild
Implementation for Shape from Polarization for Complex Scenes in the Wild (CVPR 2022)

[project website](https://chenyanglei.github.io/sfpwild/index.html) | [paper](https://arxiv.org/pdf/2112.11377.pdf)



## Dependencies and Installation
### Environment
Environment
This code is based on PyTorch. It has been tested on Ubuntu 18.04 LTS.

```
conda env create -f environment.yml
conda activate sfp_wild
```

### Data
```
bash download.sh
```
After running the script, 35GB .zip file should be downloaded.
Keep the folder structure as
```
--polar3d
 |--configs
 |--data
   |--cvpr2022_camera_ready
     |--ready_kinect3_lucid_pair_mask
 |--logs
 |--***.py
```


## Train, test and evaluation

```
bash reproduce_full_model.sh
```

## Introduction

We present a new data-driven approach with physics-based priors to scene-level normal estimation from a single polarization image. Existing shape from polarization (SfP) works mainly focus on estimating the normal of a single object rather than complex scenes in the wild. A key barrier to high-quality scene-level SfP is the lack of real-world SfP data in complex scenes. Hence, we contribute the first real-world scene-level SfP dataset with paired input polarization images and ground-truth normal maps. Then we propose a learning-based framework with a multi-head self-attention module and viewing encoding, which is designed to handle increasing polarization ambiguities caused by complex materials and non-orthographic projection in scene-level SfP. Our trained model can be generalized to far-feld outdoor scenes as the relationship between polarized light and surface normals is not affected by distance. Experimental results demonstrate that our approach significantly outperforms existing SfP models on two datasets.

<img src="figures/sfp.png" height="220px"/> 

## Citation
If you find this work useful for your research, please cite:
```
@InProceedings{Lei_2022_CVPR,
    author    = {Lei, Chenyang and Qi, Chenyang and Xie, Jiaxin and Fan, Na and Koltun, Vladlen and Chen, Qifeng},
    title     = {Shape From Polarization for Complex Scenes in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12632-12641}
}
```

## Contact
Please contact us if there is any question (Chenyang Lei, leichenyang7@gmail.com; Chenyang Qi, cqiaa@connect.ust.hk)


