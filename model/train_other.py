import os
from datetime import datetime
import warnings
from itertools import chain

from model.ProjectorLoss import ProjectorLoss

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from model.data_loader import VideoDataset
from DRNet import BaseModel
from util.get_config import get_yaml_data
from util.logger import Logger
from util.Loss import performance_fit,plcc_loss, plcc_rank_loss
from torchvision import transforms


def main(config_path):
    all_test_SRCC, all_test_PLCC= [], []
    config = get_yaml_data(config_path)
    log_path = config['log_path']
    log_path = log_path.split('.')[0]
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H:%M")
    log_path = log_path + '-' + dt_string + '.log'
    print('log to '+ log_path)
    with Logger(log_path):
        for round in range(config['round']):

            print('%d round training starts here' % int(round+1))

            print('The current model is ' + config['model_name'])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model =BaseModel(config['ops_candidates'])

            if config['use_ddp']:

                model = torch.nn.DataParallel(model, device_ids=config['gpu_ids'])

                model = model.to(device)

            else:

                model = model.to(device)

            model = model.float()

            if config['trained_model_path'] is not None:

                print('loading the pretrained model')

                model.load_state_dict(torch.load(config['trained_model_path']))

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config['lr']), weight_decay=0.0000001)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['decay_interval'], gamma=config['decay_ratio'])

            print('use loss type: {}'.format(config['loss_type']))
            if config['loss_type'] == 'plcc':
                criterion = plcc_loss
            elif config['loss_type'] == 'plcc_rank':
                criterion = plcc_rank_loss
            elif config['loss_type'] == 'L2':
                criterion = nn.MSELoss().to(device)
            elif config['loss_type'] == 'L1':
                criterion = nn.L1Loss().to(device)
            elif config['loss_type'] == 'Huberloss':
                criterion = nn.HuberLoss().to(device)

            param_num = 0
            for param in model.parameters():
                if param.requires_grad:
                   param_num += int(np.prod(param.shape))
            print('Trainable params: %.2f million' % (param_num / 1e6))

            print('The current database is ' +config['database'])

            transformations_train = transforms.Compose(
                [transforms.Resize(config['resize'], interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.RandomCrop(config['crop_size']), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transformations_test = transforms.Compose(
                [transforms.Resize(config['resize'], interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.CenterCrop(config['crop_size']), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            train_set = VideoDataset(config['saliency_path'],config['videos_path'],config['slowfast_path'], config['datainfo'], transformations_train, config['database'],
                                                config['crop_size'],"train",config['seed'])
            test_set = VideoDataset(config['saliency_path'], config['videos_path'], config['slowfast_path'],
                                                config['datainfo'], transformations_test, config['database'],
                                                config['crop_size'],"val",config['seed'])

            ## dataloader
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'],
                                                       shuffle=True, num_workers=config['num_workers'], drop_last=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                      shuffle=False, num_workers=config['num_workers'])


            best_test_criterion_SRCC = -1

            best_test = []


            print('Starting training:')

            rounds = config['round']
            epochs = config['epochs']
            for epoch in range(epochs):
                model.train()
                batch_losses = []
                batch_losses_each_disp = []
                for i, (video,fast,saliency,mos,_) in  enumerate(tqdm(train_loader,desc=f'Epoch {epoch+1}/{epochs} Round {round+1}/{rounds}',total=len(train_loader))):
                    video = video.to(device)
                    saliency = saliency.to(device)
                    fast = fast.to(device)
                    labels = mos.to(device).float()

                    outputs,_,_= model(video,saliency,fast)

                    optimizer.zero_grad()

                    loss_st = criterion(labels, outputs)

                    loss = loss_st

                    batch_losses.append(loss)

                    batch_losses_each_disp.append(loss)

                    loss.backward()

                    optimizer.step()

                avg_loss = sum(batch_losses) / (len(train_set) // config['batch_size'])
                print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
                scheduler.step()
                lr = scheduler.get_last_lr()
                print('The current learning rate is {:.08f}'.format(lr[0]))
                with torch.no_grad():
                    model.eval()
                    label = np.zeros([len(test_set)])
                    y_output = np.zeros([len(test_set)])
                    for i, (video, fast, saliency, mos,name) in enumerate(
                            tqdm(test_loader, desc="test", total=len(test_loader))):
                        video = video.to(device)
                        saliency = saliency.to(device)
                        fast = fast.to(device)
                        label[i] = mos.item()
                        outputs,_, _ = model(video, saliency, fast)
                        y_output[i] = outputs.item()
                    test_PLCC, test_SRCC, _, _ = performance_fit(label, y_output)
                    print(
                        'Epoch {} completed. The result on the  test dataset: SRCC: {:.4f}, PLCC: {:.4f}'.format(
                            epoch + 1, \
                            test_SRCC, test_PLCC))


                    if test_SRCC > best_test_criterion_SRCC:
                        print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                        best_test_criterion_SRCC = test_SRCC
                        if config['save_model_path']:
                            best_model = model.state_dict()
                            save_path = os.path.join(config['save_model_path'],config['model_name']+'_'+config['database']+'vis.pth')
                            torch.save(best_model, save_path)
                        best_test = [test_SRCC,test_PLCC]
            print('The best training result on the  test dataset SRCC: {:.4f}, PLCC: {:.4f}'.format(\
                best_test[0],best_test[1]))
            all_test_SRCC.append(best_test[0])
            all_test_PLCC.append(best_test[1])
        print(
            'The base median results test SRCC: {:.4f}, PLCC: {:.4f}'.format( \
                np.median(all_test_SRCC), np.median(all_test_PLCC)))


if __name__ == '__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config_path="/home/usr/wangweiwei/RichVQA/config/train_other.yml"
    main(config_path)