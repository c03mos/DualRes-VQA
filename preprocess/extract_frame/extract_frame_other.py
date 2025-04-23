import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio

def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap = cv2.VideoCapture(filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    # 处理帧率为0的情况
    if video_frame_rate == 0:
        # 设置一个默认的采样间隔，比如每10帧取一帧
        sampling_interval = 10
    else:
        sampling_interval = video_frame_rate

    video_read_index = 0
    frame_idx = 0
    video_length_min = 8
    last_valid_frame = None

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            last_valid_frame = frame  # 保存最后一帧有效的图像
            # 当帧率为0时，使用sampling_interval作为采样间隔
            if (video_read_index < video_length) and (frame_idx % sampling_interval == sampling_interval // 2):
                read_frame = frame
                exit_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                       '{:03d}'.format(video_read_index) + '.png'), read_frame)
                video_read_index += 1
            frame_idx += 1

    # 如果提取的帧数小于最小要求，使用最后一帧有效的图像补足
    if video_read_index < video_length_min and last_valid_frame is not None:
        for i in range(video_read_index, video_length_min):
            cv2.imwrite(os.path.join(save_folder, video_name_str,
                                   '{:03d}'.format(i) + '.png'), last_valid_frame)

    video_capture.release()
    cap.release()
    return
            
def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return

if __name__ == '__main__':
    videos_dir = 'your_path'
    filename_path = 'your_path'
    save_folder = 'your_path'
    video_names = []
    score = []
    df = pd.read_csv(filename_path)
    if 'KVQ' in  videos_dir:
        video_names = df['filename'].values
        score = df['score'].values
    else:
        video_names = df['video_path'].values
        score = df['mos'].values
    n_video = len(video_names)
    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, video_name, save_folder)
