# Data Preparation

We have successfully pre-trained and fine-tuned our VideoMAE on [Kinetics400](https://deepmind.com/research/open-source/kinetics), [Something-Something-V2](https://developer.qualcomm.com/software/ai-datasets/something-something), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) with this codebase.

- The pre-processing of **Something-Something-V2** can be summarized into 3 steps:

  1. Download the dataset from [official website](https://developer.qualcomm.com/software/ai-datasets/something-something).

  2. Preprocess the dataset by changing the video extension from `webm` to `.mp4` with the **original** height of **240px**.

  3. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). We **share** our annotation files (train.csv, val.csv, test.csv) via [Google Drive](https://drive.google.com/drive/folders/1cfA-SrPhDB9B8ZckPvnh8D5ysCjD-S_I?usp=share_link). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```

- The pre-processing of **Kinetics400** can be summarized into 3 steps:

  1. Download the dataset from [official website](https://deepmind.com/research/open-source/kinetics).

  2. Preprocess the dataset by resizing the short edge of video to **320px**. You can refer to [MMAction2 Data Benchmark](https://github.com/open-mmlab/mmaction2) for [TSN](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400-data-benchmark-8-gpus-resnet50-imagenet-pretrain-3-segments) and [SlowOnly](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly#kinetics-400-data-benchmark). <br>
**Recommend**: [OpenDataLab](https://opendatalab.com/) provides a copy of [Kinetics400](https://opendatalab.com/Kinetics-400) dataset, you can download Kinetics dataset with **short edge 320px** from [here](https://opendatalab.com/Kinetics-400).<br>

  3. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```

### Note:

1. We use [decord](https://github.com/dmlc/decord) to decode the videos **on the fly** during both pre-training and fine-tuning phases.
2. All experiments on Kinetics-400 in VideoMAE are based on [this version](https://opendatalab.com/Kinetics-400).
