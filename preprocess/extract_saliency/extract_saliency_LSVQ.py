import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from universal_module.TranSalNet_Res import TranSalNet

def preprocess_img(img_dir, channels=3,size=(288,384)):

    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    shape_r,shape_c = size
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img
    img_padded = np.array(img_padded) / 255.
    img_padded = np.expand_dims(np.transpose(img_padded, (2, 0, 1)), axis=0)
    img_padded = torch.from_numpy(img_padded)
    return img_padded


def postprocess_img(pred, org_dir):
    toPIL = transforms.ToPILImage()
    pred = toPIL(pred.squeeze())
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img.squeeze()


def get_saliency_sample(ori_img,sal_img,save_path,target_size=(224,224),grid_size=(7,7),fragment_size=(32,32)):
    assert ori_img.shape[:2] == sal_img.shape[:2]   #The shape of ori_img should be the same as the sal_img

    h,w = ori_img.shape[:2]
    if w<=target_size[0] and h<=target_size[1]:
        print("video size too small")
        img = Image.fromarray(ori_img)
        img.save(save_path)
        return ori_img
    else:
        col_width = w // grid_size[0]
        row_height = h // grid_size[1]
        img = []
        sal = []
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                start_x = col * col_width
                start_y = row * row_height
                end_x = min((col + 1) * col_width, w)
                end_y = min((row + 1) * row_height, h)
                img_piece = ori_img[start_y:end_y, start_x:end_x]
                sal_piece = sal_img[start_y:end_y, start_x:end_x]
                img.append(img_piece)
                sal.append(sal_piece)
        target_img = []
        for i in range(0,len(img)):
            cur = img[i]
            cur_s = sal[i]
            if cur.shape[0] <fragment_size[0] or cur.shape[1] <fragment_size[1]:
               cur = cv2.resize(cur, fragment_size, interpolation=cv2.INTER_LINEAR)
               target_img.append(cur)
            else:
                c_h, c_w = cur.shape[:2]
                x, y = np.unravel_index(np.argmax(cur_s), cur_s.shape)
                left = max(x - fragment_size[0]//2, 0)
                top = max(y - fragment_size[1]//2, 0)
                right = min(left + fragment_size[0], c_w)
                bottom = min(top +fragment_size[1], c_h)
                if right - left < fragment_size[0]:
                    if left == 0:
                        right = left + fragment_size[0]
                    else:
                        left = right - fragment_size[0]
                if bottom - top < fragment_size[1]:
                    if top == 0:
                        bottom = top + fragment_size[1]
                    else:
                        top = bottom - fragment_size[1]
                target_img.append(cur[top:bottom, left:right])
        output_image = Image.new('RGB', target_size, (255, 255, 255))
        for i in range(len(target_img)):
            ori_arr = target_img[i]
            ori_arr = cv2.cvtColor(ori_arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(ori_arr)
            x = (i % grid_size[0]) * fragment_size[0]
            y = (i //grid_size[0]) * fragment_size[1]
            region = (x, y, x + fragment_size[0], y + fragment_size[1])
            output_image.paste(img, region)
        output_image.save(save_path)



def get_all_saliency(model,data_dir,save_dir,video_length_read):
    folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for folder in folders:
        print('Extracting saliency samples for folder:{}'.format(folder))
        sub_folder_path = os.path.join(data_dir, folder)
        sub_folders =[d for d in os.listdir(sub_folder_path)]
        for i in tqdm(range(len(sub_folders)),desc=f'Extracting saliency samples', total = len(sub_folders)):
            folder_path = os.path.join(data_dir,sub_folder_path,sub_folders[i])
            for j in range(video_length_read):
                img_path = os.path.join(folder_path, '{:03d}'.format(int(j)) + '.png')
                save_path = os.path.join(save_dir,folder,sub_folders[i])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_dir,folder,sub_folders[i],'{:03d}'.format(int(j)) + '.png')
                get_saliency_map(model, img_path, save_path)

def saliency(model_path,device):
    model = TranSalNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def get_saliency_map(model,img_path,svae_path):
    img = preprocess_img(img_path)
    img = img.type(torch.cuda.FloatTensor).to('cuda')
    pred_saliency = model(img)
    pred_saliency = postprocess_img(pred_saliency, img_path)
    ori_img = cv2.imread(img_path)
    get_saliency_sample(ori_img,pred_saliency,svae_path)

def main(config):
    model = saliency(config.model_path, config.device)
    get_all_saliency(model,config.data_dir,config.feature_save_folder,config.frames_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='your_path/TranSalNet_Res.pth')
    parser.add_argument('--data_dir', type=str, default='your_path')
    parser.add_argument('--feature_save_folder', type=str, default='your_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--frames_size', type=int, default=8)
    config = parser.parse_args()

    main(config)