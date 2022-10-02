# VideoMAE Model Zoo

### Kinetics-400

|  Method  | Extra Data | Backbone | Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-B   |  800  | 16x5x3  | [script](scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/pretrain.sh)/[log](https://drive.google.com/file/d/1kP3_-465jCL7PRNFq1JcAghPo2BONRWY/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1JfrhN144Hdg7we213H1WxwR3lGYOlmIn/view?usp=sharing) | [script](scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/finetune.sh)/[log](https://drive.google.com/file/d/1JOJzhlCujgpsjjth0J49k5EwBNxy76xt/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/18EEgdXY9347yK3Yb28O-GxFMbk41F6Ne/view?usp=sharing)<br />(w/o repeated aug) | 80.0  | 94.4  |
| VideoMAE |  ***no***  |  ViT-B   |  800  | 16x5x3  |                        same as above                         |                             TODO                             | 81.0  | 94.8  |
| VideoMAE |  ***no***  |  ViT-B   | 1600  | 16x5x3  | [script](scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_1600/pretrain.sh)/[log](https://drive.google.com/file/d/1ftVHzzCupEGV4bCHC5JWIUsEwOEeAQcg/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing) | [script](scripts/kinetics/videomae_vit_large_patch16_224_tubemasking_ratio_0.9_epoch_1600/finetune.sh)/[log](https://drive.google.com/file/d/1fYXtL2y2ZTMxDtTRqoUOe6leVmdVI5HH/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1MzwteHH-1yuMnFb8vRBQDvngV1Zl-d3z/view?usp=sharing) | 81.5  | 95.1  |
| VideoMAE |  ***no***  |  ViT-L   | 1600  | 16x5x3  | [script](scripts/kinetics/videomae_vit_large_patch16_224_tubemasking_ratio_0.9_epoch_1600/pretrain.sh)/[log](https://drive.google.com/file/d/1X7WBzn_yG4lDWuvBMBBgrtgqDLZVHrc2/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3/view?usp=sharing) | [script](scripts/kinetics/videomae_vit_large_patch16_224_tubemasking_ratio_0.9_epoch_1600/finetune.sh)/[log](https://drive.google.com/file/d/1Doqx6zDQEMnMyPvDdz2knG385o0sZn3f/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1jX1CiqxSkCfc94y8FRW1YGHy-GNvHCuD/view?usp=sharing) | 85.2  | 96.8  |

### Something-Something V2

|  Method  | Extra Data | Backbone | Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-B   |  800  | 16x2x3  | [script](scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/pretrain.sh)/[log](https://drive.google.com/file/d/1eGS18rKvbgEJ3nbsXxokkMSwNGxxoX48/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/181hLvyrrPW2IOGA46fkxdJk0tNLIgdB2/view?usp=sharing) | [script](scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/finetune.sh)/[log](https://drive.google.com/file/d/1jYAHPcs7zt_QMPM2D_geEWoWrf3yHox8/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1xZCiaPF4w7lYmLt5o1D5tIZyDdLtJAvH/view?usp=sharing)<br />(w/o repeated aug) | 69.6  | 92.0  |
| VideoMAE |  ***no***  |  ViT-B   | 2400  | 16x2x3  | [script](scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/pretrain.sh)/[log](https://drive.google.com/file/d/148nURgfcIFBQd3IQH5YhJ9dTwNCc2jkU/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1I18dY_7rSalGL8fPWV82c0-foRUDzJJk/view?usp=sharing) | [script](scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/finetune.sh)/[log](https://drive.google.com/file/d/15TPBiUl_K2Q_9l6J41G_vf-2lovVLEHM/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1dt_59tBIyzdZd5Ecr22lTtzs_64MOZkT/view?usp=sharing) | 70.8  | 92.4  |

### Note:

- We report the results of VideoMAE finetuned with `I3D dense sampling` on **Kinetics400** and `TSN uniform sampling` on **Something-Something V2**, respectively.
- \#Frame = #input_frame x #clip x #crop.
- \#input_frame means how many frames are input for model during the test phase.
- \#crop means spatial crops (e.g., 3 for left/right/center crop).
- \#clip means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).
