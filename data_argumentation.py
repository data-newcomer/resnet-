mport torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import os
from Alexnet import *
from loss.focal import FocalLoss

# argumentation cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2

for i, (input, target) in enumerate(trainloader):

    input = input.cuda()
    target = target.cuda()
    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:
        # generate mixed sample
        """1.设定lamda的值，服从beta分布"""
        lam = np.random.beta(args.beta, args.beta)
        """2.找到两个随机样本"""
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target  # 一个batch
        target_b = target[rand_index]  # 将原有batch打乱顺序
        """3.生成剪裁区域B"""
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
        # 打乱顺序后的batch组和原有的batch组进行替换[对应id下]
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        """5.根据剪裁区域坐标框的值调整lam的值"""
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        """6.将生成的新的训练样本丢到模型中进行训练"""
        output = model(input)
        """7.按lamda值分配权重"""
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    else:
        # compute output
        output = model(input)
        loss = criterion(output, target)
# cos
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
# argumentation mixup

USE_MIXUP = True,  # 是否使用mixup方法增强数据集
MIXUP_ALPHA = 0.5,  # add mixup alpha ,用于beta分布



LOSS = FocalLoss()


def criterion(batch_x, batch_y, alpha=1.0, use_cuda=True):
    '''
    batch_x：批样本数，shape=[batch_size,channels,width,height]
    batch_y：批样本标签，shape=[batch_size]
    alpha：生成lam的beta分布参数，一般取0.5效果较好
    use_cuda：是否使用cuda

    returns：
    	mixed inputs, pairs of targets, and lam
    '''

    if alpha > 0:
        # alpha=0.5使得lam有较大概率取0或1附近
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)  # 生成打乱的batch_size索引

        # 获得混合的mixed_batchx数据，可以是同类（同张图片）混合，也可以是异类（不同图片）混合
        mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]

        """
        Example：
        假设batch_x.shape=[2,3,112,112]，batch_size=2时，
        如果index=[0,1]的话，则可看成mixed_batchx=lam*[[0,1],3,112,112]+(1-lam)*[[0,1],3,112,112]=[[0,1],3,112,112]，即为同类混合
        如果index=[1,0]的话，则可看成mixed_batchx=lam*[[0,1],3,112,112]+(1-lam)*[[1,0],3,112,112]=[batch_size,3,112,112]，即为异类混合
        """
    batch_ya, batch_yb = batch_y, batch_y[index]
    return mixed_batchx, batch_ya, batch_yb, lam


def mixup_criterion(criterion, inputs, batch_ya, batch_yb, lam):
    return lam * criterion(inputs, batch_ya) + (1 - lam) * criterion(inputs, batch_yb)


##########################################################################

#####################修改位置3：train.py文件修改代码如下######################
if torch.cuda.is_available() and device.type == "cuda":  # add
    inputs, targets = inputs.cuda(), targets.cuda()
else:
    inputs = inputs.to(device)
    targets = targets.to(device).long()

if cfg['USE_MIXUP']:
    inputs, targets_a, targets_b, lam = mixup.mixup_data(
        inputs, targets, cfg["MIXUP_ALPHA"], torch.cuda.is_available())

    # 映射为Variable
    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
    # 抽取特征，BACKBONE为粗特征抽取网络
    features = BACKBONE(inputs)
    # 抽取特征，HEAD为精细的特征抽取网络
    outputs = mixup.mixup_criterion(HEAD, features, targets_a, targets_b, lam)
    loss = mixup.mixup_criterion(LOSS, outputs, targets_a, targets_b, lam)
else:
    features = BACKBONE(inputs)
    outputs = HEAD(features, targets)
    loss = FocalLoss(outputs, labels)

