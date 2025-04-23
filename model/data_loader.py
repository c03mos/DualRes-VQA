import os
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np



def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self,saliency_dir, data_dir, data_dir_3D,filename_path, transform, database_name, crop_size,data_type,seed):
        super(VideoDataset, self).__init__()
        dataInfo= pd.read_csv(filename_path)
        random.seed(seed)
        np.random.seed(seed)
        if 'LSVQ' == database_name:
            if data_type == 'test':
                self.video_names = dataInfo['name'].tolist()
                self.score = dataInfo['mos'].tolist()
            if data_type == 'LSVQ_train':
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
                    self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
                    self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]

            elif data_type == 'LSVQ_val':
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
                self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
                self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

            elif data_type == 'LSVQ_test':
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
                self.video_names = dataInfo['name'].tolist()
                self.score = dataInfo['mos'].tolist()

            elif data_type == 'LSVQ_test_1080p':
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
                dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                       encoding="utf-8-sig")
                self.video_names = dataInfo['name'].tolist()
                self.score = dataInfo['mos'].tolist()
        elif 'KVQ' == database_name:
               self.video_names = dataInfo['video_path'].tolist()
               self.score = dataInfo['mos'].tolist()
        elif 'MWV' == database_name:
               self.video_names = dataInfo['video_path'].tolist()
               self.score = dataInfo['mos'].tolist()
        else:
            dataInfo.columns = ['video_path', 'mos']
            length = dataInfo.shape[0]
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if data_type == 'train':
                self.video_names =dataInfo.iloc[train_index]['video_path'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif data_type == 'val':
                self.video_names = dataInfo.iloc[val_index]['video_path'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif data_type == 'test':
                self.video_names = dataInfo.iloc[test_index]['video_path'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()
        self.saliency_dir=saliency_dir
        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if 'LSVQ' in self.database_name :
            video_name = str(self.video_names[idx])
        elif'Youtube' in self.database_name:
            video_name = str(self.video_names[idx][:-4])
        else:
            video_name = str(self.video_names[idx])
            video_name = video_name.split('.')[0]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name)

        saliency_name=os.path.join(self.saliency_dir,video_name)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        saliency_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame
            imge_name1 = os.path.join(saliency_name, '{:03d}'.format(int(1 * i)) + '.png')
            read_frame1 = Image.open(imge_name1)
            read_frame1 = read_frame1.convert('RGB')
            read_frame1 = self.transform(read_frame1)
            saliency_video[i] = read_frame1

        feature_folder_name = os.path.join(self.data_dir_3D, video_name)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            i_index = i
            if 'Youtube' in  self.database_name:
                feature_3D = np.load(os.path.join(feature_folder_name.split('.')[0], 'feature_' + str(i_index) + '_fast_feature.npy'))
            else:
                feature_3D = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D


        return transformed_video, transformed_feature, saliency_video,video_score,video_name