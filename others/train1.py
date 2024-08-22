import os
import argparse
import datetime
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.self_dataset import self_dataset
from dataset.dataset_OneChannel import dataset_oneChannel
from logger.normal_logger import normal_logger
from logger.tensorboard_logger import tensorboard_logger
from module.one_stage_model import one_stage_model
from module.one_stage_loss import RegLoss, FastFocalLoss
from module.Center_HM_labels import get_HM_label, get_OneStage_lidar_annotation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, nargs='*', default=[0], help='specify GPU device Id')
    parser.add_argument('--ddp', action='store_true', help='use pytorch torch.nn.parallel.DistributedDataParallel')

    parser.add_argument('--save_dir', type=str, default='./work_dir', help='log path')
    parser.add_argument('--exp_prefix', type=str, default=None, help='experiment name prefix')
    parser.add_argument('--resume_from', type=str, default=None, help='resume from dir')
    # '/home/zhangq/Desktop/zhangq/HM_RGB_Net/work_dir/20220701_16:58:29'

    parser.add_argument('--dataset_path', type=str, default='/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train', help='dataset path')
    parser.add_argument('--dataset_val_path', type=str, default='/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_val', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--num_workers', type=int, default=1, help='batch Size during training')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')

    parser.add_argument('--epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--eval_per_epoch', type=int, default=3, help='how many epoches is run before one valid_loop')
    parser.add_argument('--save_per_epoch', type=int, default=5, help='how many epoches is run before save')

    args = parser.parse_args()
    return args

class HM_RGB_net():
    def __init__(self):
        self.Center_HM_Model = one_stage_model()
        self.FocalLoss = FastFocalLoss()


def train(args):
    '''SELECT DEVICE'''
    device = torch.device("cuda:{}".format(args.gpu[0]))

    '''CREATE DIR'''
    if args.resume_from is None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            print("create {}.".format(args.save_dir))
        if args.exp_prefix is None:
            exp_name = str(datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
        else:
            exp_name = args.exp_prefix + "_" + str(datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
        exp_dir = os.path.join(args.save_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
            print("create {}.".format(exp_dir))
        exp_checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        if not os.path.exists(exp_checkpoint_dir):
            os.mkdir(exp_checkpoint_dir)
            print("create {}.".format(exp_checkpoint_dir))
        exp_logs_dir = os.path.join(exp_dir, "logs")
        if not os.path.exists(exp_logs_dir):
            os.mkdir(exp_logs_dir)
            print("create {}.".format(exp_logs_dir))
        exp_log_dir = os.path.join(exp_logs_dir, str(datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')))
        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)
            print("create {}.".format(exp_log_dir))
    else:
        exp_dir = args.resume_from
        exp_checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        exp_logs_dir = os.path.join(exp_dir, 'logs')
        assert os.path.exists(exp_checkpoint_dir)
        assert os.path.exists(exp_logs_dir)
        exp_log_dir = os.path.join(exp_logs_dir, str(datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')))
        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)
            print("create {}.".format(exp_log_dir))

    '''SAVE SCRIPT'''
    if not os.path.exists(os.path.join(exp_dir, os.path.basename(__file__))):
        shutil.copy(os.path.basename(__file__), exp_dir)

    '''CREATE LOGGER'''
    logger = normal_logger(exp_log_dir, print_on=False)
    tensorboard_logger_train = tensorboard_logger(exp_log_dir, isTrain=True)
    tensorboard_logger_val = tensorboard_logger(exp_log_dir, isTrain=False)

    logger.log('PARAMETER...', print_on=True)
    logger.log(args, print_on=True)

    '''LOAD DATASET'''
    # dataset_train = self_dataset(args.dataset_path)
    # dataset_val = self_dataset(args.dataset_val_path)
    dataset_train = dataset_oneChannel(args.dataset_path)
    dataset_val = dataset_oneChannel(args.dataset_val_path)

    logger.log('LOAD DATASET...', print_on=True)
    logger.log('The length of dataset_train = {}'.format(len(dataset_train)), print_on=True)
    logger.log('The length of dataset_val = {}'.format(len(dataset_val)), print_on=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)

    '''BUILD NET'''
    net = HM_RGB_net()
    Center_HM_Model = net.Center_HM_Model.to(device)
    FocalLoss = net.FocalLoss.to(device)

    if args.resume_from is not None:
        # training resume
        checkpoint = torch.load(os.path.join(exp_checkpoint_dir, 'epoch10.pth'))
        start_epoch = checkpoint['epoch'] + 1  # start from next epoch
        Center_HM_Model.load_state_dict(checkpoint['Center_HM_Model'])
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                Center_HM_Model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0
            )
        else:
            optimizer = torch.optim.SGD(Center_HM_Model.parameters(), lr=0.01, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        scheduler.load_state_dict(checkpoint['scheduler'])
        step_global = checkpoint['step_global']
        loss_min = checkpoint['loss_min']

        logger.log('Resume model from {}'.format(os.path.join(exp_checkpoint_dir, 'latest.pth')))
    else:
        # training from scratch
        Center_HM_Model = Center_HM_Model.apply(weights_init)
        start_epoch = 0
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                Center_HM_Model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0
            )
        else:
            optimizer = torch.optim.SGD(Center_HM_Model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        step_global = 0
        loss_min = np.Inf

        logger.log('Training from scratch')

    logger.log('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.log('>>>> Train Epoch {}/{}'.format(epoch+1, args.epoch), print_on=True)

        learning_rate_current = optimizer.state_dict()['param_groups'][0]['lr']
        tensorboard_logger_train.log_scalar('lr', learning_rate_current, epoch + 1)
        logger.log('learning_rate = {}'.format(learning_rate_current), print_on=True)

        Center_HM_Model.train()
        loss_sum = 0
        for step, (One_Stage_HM, rgb_img, dynamic_heatmap, static_heatmap) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9):
            optimizer.zero_grad()

            One_Stage_HM = One_Stage_HM.float().to(device)    # (B, 1, 128, 128)
            # rgb_img = rgb_img.float().to(device)
            dynamic_heatmap = dynamic_heatmap.float().to(device)
            static_heatmap = static_heatmap.float().to(device)

            # HM_x, HM_y, HM_l, HM_w, HM_alpha = get_OneStage_lidar_annotation(lidar_annotation)
            # One_Stage_HM = get_HM_label(HM_x, HM_y, HM_l, HM_w, HM_alpha)
            # One_Stage_HM = One_Stage_HM.float().to(device)


            pred = Center_HM_Model(dynamic_heatmap, static_heatmap)  # (B, 1, 128, 128)

            # pred_nd = pred.cpu().detach().numpy()
            # One_Stage_HM_nd = One_Stage_HM.cpu().numpy()
            # ax1 = plt.subplot(1, 2, 1)
            # plt.pcolor(pred_nd[0][0])
            # plt.axis('off')
            # # plt.show()
            # # hm = pred_nd[0][0] - One_Stage_HM_nd[0][0]
            # # print('****************************hm{:.4f}*********************'.format(hm.sum()))
            # ax2 = plt.subplot(1, 2, 2)
            # plt.pcolor(One_Stage_HM_nd[0][0])
            # plt.axis('off')
            # # plt.show() , bbox_inches='tight', pad_inches=-0.1
            # save_name = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/' + str(step) + '_train.jpg'
            # plt.savefig(save_name)

            HM_loss = FocalLoss(pred, One_Stage_HM)
            logger.log("step {}/{}: loss = {}".format(step + 1, len(dataloader_train),  HM_loss))
            tensorboard_logger_train.log_scalar('loss', HM_loss, step_global + 1)

            HM_loss.backward()
            optimizer.step()

            # print('HM_loss:', HM_loss)

            loss_sum += HM_loss
            step_global += 1

        loss_mean = loss_sum / len(dataloader_train)
        logger.log("loss_mean = {}".format(loss_mean), print_on=True)
        tensorboard_logger_train.log_scalar('loss_mean', loss_mean, epoch + 1)


        scheduler.step()

        # save and build link
        if args.save_per_epoch > 0:
            if (epoch - start_epoch + 1) % args.save_per_epoch == 0:  # whether need to be save
                is_update = False
                if loss_mean < loss_min:
                    loss_min = loss_mean
                    is_update = not is_update
                state = {
                    'epoch': epoch,
                    'step_global': step_global,
                    'Center_HM_Model': Center_HM_Model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss_min': loss_min,
                }

                # save epoch{}.pth
                torch.save(state, os.path.join(exp_checkpoint_dir, 'epoch{}.pth'.format(epoch + 1)))

                # build latest.pth link
                try:
                    os.remove(os.path.join(BASE_DIR, exp_checkpoint_dir, 'latest.pth'))
                except:
                    pass
                os.symlink(
                    os.path.join(BASE_DIR, exp_checkpoint_dir, 'epoch{}.pth'.format(epoch + 1)).format(epoch + 1),
                    os.path.join(BASE_DIR, exp_checkpoint_dir, 'latest.pth')
                )

                if is_update:  # whether new loss_min occur
                    try:
                        os.remove(os.path.join(BASE_DIR, exp_checkpoint_dir, 'best.pth'))
                    except:
                        pass
                    # build best.pth link
                    os.symlink(
                        os.path.join(BASE_DIR, exp_checkpoint_dir, 'epoch{}.pth'.format(epoch + 1)).format(epoch + 1),
                        os.path.join(BASE_DIR, exp_checkpoint_dir, 'best.pth')
                    )

        # validate
        if args.eval_per_epoch > 0:
            if (epoch - start_epoch + 1) % args.eval_per_epoch == 0:
                logger.log('>>>> Val Epoch', print_on=True)

                Center_HM_Model = Center_HM_Model.eval()
                loss_sum_val = 0
                with torch.no_grad():
                    for step, (One_Stage_HM, rgb_img, dynamic_heatmap, static_heatmap) in tqdm(enumerate(dataloader_val), total=len(dataloader_val), smoothing=0.9):

                        One_Stage_HM = One_Stage_HM.float().to(device)  # (B, 1, 128, 128)
                        dynamic_heatmap = dynamic_heatmap.float().to(device)
                        static_heatmap = static_heatmap.float().to(device)

                        pred = Center_HM_Model(dynamic_heatmap, static_heatmap)  # (B, 1, 128, 128)

                        # pred_nd = pred.cpu().detach().numpy()
                        # One_Stage_HM_nd = One_Stage_HM.cpu().numpy()
                        # ax1 = plt.subplot(1, 2, 1)
                        # plt.pcolor(pred_nd[0][0])
                        # plt.axis('off')
                        # # plt.show()
                        # # hm = pred_nd[0][0] - One_Stage_HM_nd[0][0]
                        # # print('****************************hm{:.4f}*********************'.format(hm.sum()))
                        # ax2 = plt.subplot(1, 2, 2)
                        # plt.pcolor(One_Stage_HM_nd[0][0])
                        # plt.axis('off')
                        # # plt.show() , bbox_inches='tight', pad_inches=-0.1
                        # save_name = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/' + str(step) + '_val.jpg'
                        # plt.savefig(save_name)

                        HM_loss = FocalLoss(pred, One_Stage_HM)
                        logger.log("step {}/{}: loss = {}".format(step + 1, len(dataloader_val), HM_loss))
                        loss_sum_val += HM_loss
                    loss_mean_val = loss_sum_val / len(dataloader_val)

                logger.log("loss_mean_val = {}".format(loss_mean_val), print_on=True)
                tensorboard_logger_val.log_scalar('loss_mean', loss_mean_val, epoch + 1)

    print('train done')


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(model.weight.data)
        # torch.nn.init.constant_(model.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(model.weight.data)
        # torch.nn.init.constant_(model.bias.data, 0.0)

def train_parallel(args):
    pass

def train_ddp_parallel(args):
    pass

def main():
    args = get_args()
    if len(args.gpu) > 1:
        if args.ddp:
            train_ddp_parallel(args)
        else:
            train_parallel(args)
    else:
        train(args)

if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()

