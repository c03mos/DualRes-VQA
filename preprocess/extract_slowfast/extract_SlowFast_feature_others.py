# -*- coding: utf-8 -*-
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self,database_name,data_dir, filename_path, transform, resize, num_frame):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()
        self.database_name = database_name
        df = pd.read_csv(filename_path)
        self.video_names = df['video_path'].values
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)
        self.num_frame = num_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        filename = os.path.join(self.videos_dir, video_name)
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)
        video_channel = 3
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        if video_frame_rate == 0:
            video_clip_min = 10
            if video_length > video_clip_min:
                video_clip = video_clip_min
            else:
                video_clip = video_length
        else:
            video_clip_min = int(video_length / video_frame_rate)
            video_clip = int(video_length / video_frame_rate)
        if self.database_name == 'KoNViD-1k' or self.database_name == 'LIVEYTGaming':
            video_clip_min = 8
        elif self.database_name == 'LiveVQC' or self.database_name == 'CVD2014':
            video_clip_min = 10
        elif self.database_name == 'youtube_ugc':
            video_clip_min = 10

        video_length_clip = self.num_frame

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_name

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list
class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)

        return slow_feature, fast_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='youtube_ugc')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_frame', type=int, default=32)
    parser.add_argument('--videos_dir', type=str, default='your_path')
    parser.add_argument('--datainfo', type=str, default='your_path')
    parser.add_argument('--feature_save_folder', type=str, default='your_path')
    parser.add_argument('--error_log_file', type=str, default='error_videos.log')

    config = parser.parse_args()

    def main(config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = slowfast()
        model = model.to(device)

        resize = config.resize
        transformations = transforms.Compose([
            transforms.Resize([resize, resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

        videos_dir = config.videos_dir
        datainfo = config.datainfo
        trainset = VideoDataset_NR_SlowFast_feature(config.database, videos_dir, datainfo, transformations, resize, config.num_frame)

        # Dataloader
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False, num_workers=config.num_workers
        )

        # Log file for errors
        error_log_file = config.error_log_file
        with open(error_log_file, 'w') as f:
            f.write("Error Log: Videos that failed to process\n")

        # Validation loop
        with torch.no_grad():
            model.eval()
            for i, (video, video_name) in enumerate(train_loader):
                video_name = video_name[0]
                try:
                    print(f'Processing video: {video_name} --- {i+1}/{len(train_loader)}')
                    video_name_clean = video_name.split('.')[0]
                    if video_name_clean == '287':
                        break
                    if not os.path.exists(config.feature_save_folder + video_name_clean):
                        os.makedirs(config.feature_save_folder + video_name_clean)
                    for idx, ele in enumerate(video):
                        ele = ele.permute(0, 2, 1, 3, 4)
                        inputs = pack_pathway_output(ele, device)
                        _, fast_feature = model(inputs)
                        np.save(
                            config.feature_save_folder + video_name_clean + '/' + 'feature_' + str(idx) + '_fast_feature',
                            fast_feature.to('cpu').numpy()
                        )
                except Exception as e:
                    # Log the error and video path
                    error_msg = f"Error processing video {video_name}: {str(e)}\n"
                    print(error_msg)
                    with open(error_log_file, 'a') as f:
                        f.write(error_msg)
                    continue

    main(config)