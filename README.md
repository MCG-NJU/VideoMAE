# Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training [[Arxiv]](https://arxiv.org/abs/2203.12602)

![VideoMAE Framework](figs/videomae.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data-1/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=videomae-masked-autoencoders-are-data-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data-1/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=videomae-masked-autoencoders-are-data-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data-1/self-supervised-action-recognition-on-ucf101)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-ucf101?p=videomae-masked-autoencoders-are-data-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-masked-autoencoders-are-data-1/self-supervised-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-hmdb51?p=videomae-masked-autoencoders-are-data-1)


> [**VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**](https://arxiv.org/abs/2203.12602)<br>
> [Zhan Tong](https://github.com/yztongzhan), [Yibing Song](https://ybsong00.github.io/), [Jue Wang](https://juewang725.github.io/), [Limin Wang](http://wanglimin.github.io/)<br>Nanjing University, Tencent AI Lab

## 📰 News

**[2022.4.24]**  Code and pre-trained models are available now! Please give a star⭐️ for our best efforts.😆<br>**[2022.4.15]** The **[LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE)** of this project has been upgraded to CC-BY-NC 4.0. <br>
**[2022.3.24]** ~~Code and pre-trained models will be released here.~~ Welcome to **watch** this repository for the latest updates.

## ✨ Highlights

### 🔥 Masked Video Modeling for Video Pre-Training

VideoMAE performs the task of masked video modeling for video pre-training. We propose the **extremely high** masking ratio (90%-95%) and **tube masking** strategy to create a challenging task for self-supervised video pre-training.

### ⚡️ A Simple, Efficient and Strong Baseline in SSVP

VideoMAE uses the simple masked autoencoder and **plain ViT** backbone to perform video self-supervised learning. Due to the extremely high masking ratio, the pre-training time of VideoMAE is **much shorter** than contrastive learning methods (**3.2x** speedup). VideoMAE can serve as **a simple but strong baseline** for future research in self-supervised video pre-training.

### 😮 High performance, but NO extra data required

VideoMAE works well for video datasets of different scales and can achieve **84.7%** on Kinects-400, **75.3%** on Something-Something V2, **90.8%** on UCF101, and **61.1%** on HMDB51. To our best knowledge, VideoMAE is the **first** to achieve the state-of-the-art performance on these four popular benchmarks with the **vanilla ViT** backbones while **doesn't need** any extra data or pre-trained models.

## 🚀 Main Results

### ✨ Something-Something V2

|  Method  | Extra Data | Backbone | Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-B   |         16x2x3         | 70.6  | 92.6  |
| VideoMAE |  ***no***  |  ViT-L   |         16x2x3         | 74.2  | 94.7  |
| VideoMAE |  ***no***  |  ViT-L   |         32x1x3         | 75.3  | 95.2  |

### ✨ Kinetics-400

|  Method  |  Extra Data  | Backbone | Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :----------: | :------: | :--------------------: | :---: | :---: |
| VideoMAE |   ***no***   |  ViT-B   |         16x5x3         | 80.9  | 94.7  |
| VideoMAE |   ***no***   |  ViT-L   |         16x5x3         | 84.7  | 96.5  |
| VideoMAE | Kinetics-700 |  ViT-L   |         16x5x3         | 85.8  | 96.8  |

### ✨ UCF101 & HMDB51

|  Method  |  Extra Data  | Backbone | UCF101 | HMDB51 |
| :------: | :----------: | :------: | :----: | :----: |
| VideoMAE |   ***no***   |  ViT-B   |  90.8  |  61.1  |
| VideoMAE | Kinetics-400 |  ViT-B   |  96.1  |  73.3  |

## 🔨 Installation

Please follow the instructions in [INSTALL.md](INSTALL.md).

## ➡️ Data Preparation

Please follow the instructions in [DATASET.md](DATASET.md) for data preparation.

## 🔄 Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

## ⤴️ Fine-tuning with pre-trained models

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

## 📍Model Zoo

We provide pre-trained and fine-tuned models in [MODEL_ZOO.md](MODEL_ZOO.md).

## 👀 Visualization

We provide the script for visualization in [`vis.sh`](vis.sh).  Colab notebook for better visualization is coming soon.

## ☎️ Contact 

Zhan Tong: tongzhan@smail.nju.edu.cn

## 👍 Acknowledgements

Thanks to [Ziteng Gao](https://sebgao.github.io/), Lei Chen and [Chongjian Ge](https://chongjiange.github.io/) for their kindly support.<br>
This project is built upon [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit). Thanks to the contributors of these great codebases.

## 🔒 License

This project is released under the CC-BY-NC 4.0 license as found in the [LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) file.

## ✏️ Citation

If you think this project is helpful, please feel free to give a star⭐️ and cite our paper:

```
@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```
