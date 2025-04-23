# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import cv2
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn


def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p=False):
        super(VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()
        if is_test_1080p:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
        else:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

        filename = os.path.join(self.videos_dir, video_name + '.mp4')

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_clip = int(video_length / video_frame_rate)

        video_clip_min = 8

        video_length_clip = 32

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

        return transformed_video_all, video_score, video_name


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

# keep fast pathway
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


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = slowfast()

    model = model.to(device)

    resize = config.resize

    ## training data
    videos_dir = 'your_path'
    datainfo_train = 'your_path'
    datainfo_test = 'your_path'
    datainfo_test_1080p = 'your_path'
    transformations_test = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), \
                                               transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                                    std=[0.225, 0.225, 0.225])])
    trainset = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_train, transformations_test, resize)
    testset = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize)
    testset_1080p = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_test_1080p, transformations_test, resize,
                                                          is_test_1080p=True)

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                               shuffle=False, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
                                                    shuffle=False, num_workers=config.num_workers)

    # do validation after each epoch
    with torch.no_grad():
        model.eval()
        for i, (video, mos, video_name) in enumerate(train_loader):
            video_name = video_name[0]
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)
            print("start extract {}th/{}th video:{}".format(i, len(test_loader), video_name))
            for idx, ele in enumerate(video):
                ele = ele.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature',
                        slow_feature.to('cpu').numpy())
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature',
                        fast_feature.to('cpu').numpy())

        for i, (video, mos, video_name) in enumerate(test_loader):
            video_name = video_name[0]
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)
            print("start extract {}th/{}th video:{}".format(i, len(test_loader), video_name))
            for idx, ele in enumerate(video):
                ele = ele.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature',
                        slow_feature.to('cpu').numpy())
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature',
                        fast_feature.to('cpu').numpy())
        for i, (video, mos, video_name) in enumerate(test_loader_1080p):
            video_name = video_name[0]
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)
            print("start extract {}th/{}th video:{}".format(i, len(test_loader), video_name))
            for idx, ele in enumerate(video):
                ele = ele.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature',
                        slow_feature.to('cpu').numpy())
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature',
                        fast_feature.to('cpu').numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='your_path')

    config = parser.parse_args()

    main(config)