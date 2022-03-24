# Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training [[Arxiv]](https://arxiv.org/abs/2203.12602)

![VideoMAE Framework](figs/videomae.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=videomae-masked-autoencoders-are-data)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=videomae-masked-autoencoders-are-data)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data/self-supervised-action-recognition-on-ucf101)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-ucf101?p=videomae-masked-autoencoders-are-data)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data/self-supervised-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-hmdb51?p=videomae-masked-autoencoders-are-data)


> [**VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**](https://arxiv.org/abs/2203.12602)<br>
> [Zhan Tong](https://github.com/yztongzhan), [Yibing Song](https://ybsong00.github.io/), [Jue Wang](https://juewang725.github.io/), [Limin Wang](http://wanglimin.github.io/)<br>Nanjing University, Tencent AI Lab

## News

Code and pre-trained models will be released here. Welcome to **watch** this repository for the latest updates.



## Main Results

### Something-Something V2

|  Method  | Extra Data | Backbone | Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-B   |         16x2x3         | 70.3  | 92.7  |
| VideoMAE |  ***no***  |  ViT-L   |         16x2x3         |       | 94.7  |
| VideoMAE |  ***no***  |  ViT-L   |         32x1x3         | 75.3  | 95.2  |



### Kinetics-400

|  Method  |  Extra Data  | Backbone | Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :----------: | :------: | :--------------------: | :---: | :---: |
| VideoMAE |   ***no***   |  ViT-B   |         16x5x3         | 80.7  | 94.7  |
| VideoMAE |   ***no***   |  ViT-L   |         16x5x3         | 83.9  | 96.3  |
| VideoMAE | Kinetics-700 |  ViT-L   |         16x5x3         | 85.8  | 96.8  |



### UCF101 & HMDB51

|  Method  |  Extra Data  | Backbone | UCF101 | HMDB51 |
| :------: | :----------: | :------: | :----: | :----: |
| VideoMAE |   ***no***   |  ViT-B   |  90.8  |  61.1  |
| VideoMAE | Kinetics-400 |  ViT-B   |  96.1  |  73.3  |



## Contact 

tongzhan@smail.nju.edu.cn

## Citation

```
@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```

