import csv
import pandas as pd
import scipy.io as scio
def read_float_with_comma(num):
    return float(num.replace(",", "."))
def mat2csv(mat_path,save_path):
    video_names =[]
    score =[]
    if 'YouTube_UGC' in mat_path:
        df = pd.read_excel(mat_path, sheet_name='diff2')
        video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in df['vid'].astype(str).tolist()]
        score = df['MOS full'].tolist()
    elif 'LIVEYTGaming'in mat_path:
        mat = scio.loadmat(mat_path)
        index_all = mat['index'][0]
        for i in index_all:
            video_names.append(mat['video_list'][i][0][0] + '.mp4')
            score.append(mat['MOS'][i][0])
    elif 'LSVQ' in mat_path:
        mat = scio.loadmat(mat_path)
        index_all = mat['index'][0]
        for i in index_all:
            video_names.append(mat['video_list'][i][0][0] + '.mp4')
            score.append(mat['MOS'][i][0])
    elif  'LiveVQC'in mat_path:
        mat = scio.loadmat(mat_path)
        dataInfo = pd.DataFrame(mat['video_list'])
        dataInfo['MOS'] = mat['mos']
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        video_names = dataInfo['file_names'].tolist()
        score = dataInfo['MOS'].tolist()
    elif 'LBVD'in mat_path:
        mat = scio.loadmat(mat_path)
        index_all = mat['index'][0]
        for i in index_all:
            video_names.append(mat['video_names'][i][0][0])
            score.append(mat['scores'][i][0])
    elif 'LIVE-Qualcomm'in mat_path:
        mat = scio.loadmat(mat_path)
        dataInfo = pd.DataFrame(mat['qualcommVideoData'][0][0][0])
        dataInfo['MOS'] = mat['qualcommSubjectiveData'][0][0][0]
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
        video_names = dataInfo['file_names'].tolist()
        score = dataInfo['MOS'].tolist()
    elif 'CVD2014' in mat_path:
        file_names = []
        mos = []
        openfile = open(mat_path, 'r', newline='')
        lines = csv.DictReader(openfile, delimiter=';')
        for line in lines:
            if len(line['File_name']) > 0:
                file_names.append(line['File_name'])
            if len(line['Realignment MOS']) > 0:
                mos.append(read_float_with_comma(line['Realignment MOS']))
        dataInfo = pd.DataFrame(file_names)
        dataInfo['MOS'] = mos
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'] + ".avi"
        video_names = dataInfo['file_names'].tolist()
        score = dataInfo['MOS'].tolist()
    elif 'KoNViD-1k' in mat_path:
        mat = scio.loadmat(mat_path)
        index_all = mat['index'][0]
        for i in index_all:
            video_names.append(mat['video_names'][i][0][0].split('_')[0] + '.mp4')
            score.append(mat['scores'][i][0])
    elif 'KVQ'in mat_path:
        df = pd.read_csv(mat_path)
        video_names = df['filename'].values
        score = df['score'].values
    elif 'MWV' in mat_path:
        df = pd.read_csv(mat_path)
        video_names = df['Video Name'].values
        score = df['MOS'].values
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video_path', 'mos'])
        for name, s in zip(video_names, score):
            writer.writerow([name, s])

mat_path = '/home/usr/wangweiwei/RVQA/data/MWV_train.csv'
save_path = '/home/usr/wangweiwei/RVQA/data/MWV_train_data.csv'
mat2csv(mat_path, save_path)
