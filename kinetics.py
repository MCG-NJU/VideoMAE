import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import cv2
import random

class VideoClsDataset(Dataset):

    def __init__(
        self,
        anno_path,
        data_path,
        mode='train',
        clip_len=16, # 8 ->16
        frame_sample_rate=2,
        crop_size=224,
        short_side_size=224,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        test_num_segment=1, # 10->1
        test_num_crop=1, #3->1
        args=None
    ):

        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        #validation transformer
        if self.mode == "validation":
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size),
                video_transforms.CenterCrop(self.crop_size),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        #test transformer
        if self.mode == "test":
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size),   
                video_transforms.CenterCrop(self.crop_size),     
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.dataset_samples)
    #train augmentation
    def _aug_frame(self, buffer, args):
        # resize
        buffer = self._resize(buffer, self.new_height, self.new_width)

        # random crop
        buffer = self._random_crop(buffer, self.crop_size)

        # horizontal flip
        buffer = self._random_horizontal_flip(buffer)

        # convert to tensor
        buffer = volume_transforms.ClipToTensor()(buffer)

        # normalize
        buffer = video_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(buffer)

        return buffer


    def _resize(self, buffer, h, w):
        return np.stack([cv2.resize(f, (w, h)) for f in buffer], axis=0)

    def _random_crop(self, buffer, size):
        h, w = buffer.shape[1], buffer.shape[2]
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return buffer[:, top:top+size, left:left+size, :]

    def _random_horizontal_flip(self, buffer):
        if random.random() < 0.5:
            return buffer[:, :, ::-1, :]
        return buffer

    #get item
    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        label = int(self.label_array[index])
        name = os.path.basename(sample).split('.')[0]
        buffer = self.safe_load_video(sample)
        if buffer is None:
            raise RuntimeError(f"Failed loading video {sample}")

        #train
        if self.mode == "train":

            total_frames = buffer.shape[0]
            num_frames_to_sample = self.clip_len  

            # Temporal Segment Sampling
            num_segments = self.num_segment       
            frames_per_segment = num_frames_to_sample // num_segments  

            
            if total_frames < num_frames_to_sample:
                indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

            else:
                indices = []
                for i in range(num_segments):
                    seg_start = i * (total_frames // num_segments)
                    seg_end = (i + 1) * (total_frames // num_segments) if i < num_segments - 1 else total_frames

                    seg_len = seg_end - seg_start

                    if seg_len < frames_per_segment:
                        seg_idx = np.linspace(seg_start, seg_end - 1, frames_per_segment, dtype=int)
                    else:
                        start_idx = np.random.randint(seg_start, seg_end - frames_per_segment + 1)
                        seg_idx = np.arange(start_idx, start_idx + frames_per_segment)

                    indices.extend(seg_idx)

                if len(indices) > num_frames_to_sample:
                    indices = indices[:num_frames_to_sample]
                elif len(indices) < num_frames_to_sample:
                    indices = np.pad(indices, (0, num_frames_to_sample - len(indices)), 'wrap')

            buffer = buffer[indices]

            # Augmentation
            buffer = self._aug_frame(buffer, self.args)

            return buffer, label, index, {}



        # validation
        if self.mode == "validation":
            buffer = self.temporal_sampling(buffer)
            buffer = self.data_transform(buffer)
            name = os.path.basename(sample).split('.')[0]
            return buffer, label, name

        # test
        if self.mode == "test":

            total_frames = buffer.shape[0]
            num_frames = self.clip_len

            if total_frames < num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            else:
                start = (total_frames - num_frames) // 2
                indices = np.arange(start, start + num_frames)

            clip = buffer[indices]  # (T, H, W, C)

            # transform
            clip = self.data_transform(clip)  # (C, T, H, W)

            if clip.shape[0] != 3:
                clip = clip.permute(3, 0, 1, 2)

            name = os.path.basename(sample).split('.')[0]

            return clip, label, name, 0, 0



    #safe video loading
    def safe_load_video(self, sample):
        path = os.path.join(self.data_path, sample)
        path = path.replace("\\", "/")

        if not os.path.exists(path):
            print("Missing:", path)
            return None

        try:
            vr = VideoReader(path, ctx=cpu(0))
        except:
            print("Decord failed:", path)
            return None

        if len(vr) < self.clip_len:
            return None

        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames

    #sampeling
    def temporal_sampling(self, buffer, segment_id=0):
        total = buffer.shape[0]
        segment_length = total // self.test_num_segment
        start = segment_id * segment_length
        end = min(start + segment_length, total)
        indices = np.linspace(start, end - 1, self.clip_len).astype(int)
        return buffer[indices]

    def spatial_sampling_test(self, clip):
        clip = torch.from_numpy(clip)
        clip = clip.permute(3, 0, 1, 2).float()
        clip = torch.nn.functional.interpolate(
            clip,
            size=(self.short_side_size, self.short_side_size),
            mode='bilinear',
            align_corners=False
        )
        clip = clip / 255.0

        return clip






def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
