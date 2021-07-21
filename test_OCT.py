import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize
from data.csv_dataset import ImageDataset
from utils.utils import calc_kappa
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, hamming_loss

import time
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL = "scnet50"
LOSS = 'bceloss'

model_path = './model/OCT/2021_05_20+scnet50++500+0.001+0.0001+bceloss.pth'
print("Test OCT:", model_path)

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

IMAGE_SIZE = 224  # 224 for resnet, 299 for inception

AVERAGE = 'weighted'

cols = ['视网膜内液性暗腔', '视网膜下积液', 'RPE脱离', 'RPE下高反射病灶', '视网膜内或视网膜下高反射病灶', '尖锐的RPED峰',
        '双层征', '多发性RPED', 'RPED切迹', '视网膜内高反射硬性渗出']
classCount = len(cols)
samples_per_cls = [445, 543, 808, 109, 1090, 23, 241, 88, 21, 158]

# 训练的df 路径
data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/OCT/'


def test(model, test_loader, criterion):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(test_loader, desc='\r', ncols=100)
    loss_val = 0
    loss_valnorm = 0
    for batch_idx, (inputs, target) in enumerate(tbar):
        target = target.float()  # 多分类用.float()
        data, target = inputs.cuda(), target.cuda()  # leftImg.cuda(),target.cuda()

        output = model(data)

        # start magic loss
        if LOSS == 'magic_loss':
            beta = 0.9999

            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * classCount

            weights = torch.tensor(weights).float()
            weights = weights.unsqueeze(0)
            weights = weights.repeat(target.shape[0], 1).cuda() * target.cuda()

            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, classCount)
            # loss = F.binary_cross_entropy(output, target, weight=weights)
            criterion = nn.BCELoss(reduction='mean', weight=weights)
            loss = criterion(output, target)
        # end magic loss
        else:
            loss = criterion(output, target)

        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        tbar.set_description('Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(inputs), len(test_loader.dataset),
            100. * batch_idx / len(test_loader), loss.item()))
        loss_val += loss.item()
        loss_valnorm += 1

    out_loss = loss_val / loss_valnorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)
    # kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)  # 多分类卡0.5的阈值
    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true, y_pred)

    avg = (f1 + recall + precision + auroc) / 4.0
    tbar.close()
    print(f1, auroc, recall, precision, acc, avg, hamming)
    return avg, out_loss


def main():
    if MODEL == "resnet18":
        model = models.resnet18(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())  # 多分类用sigmoid
    elif MODEL == "resnet34":
        model = models.resnet34(pretrained=False)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "resnet50":
        model = models.resnet50(pretrained=False)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "resnest50":
        from resnest.torch import resnest50
        model = resnest50(pretrained=False)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "inceptionv3":
        model = models.inception_v3(pretrained=False, aux_logits=False)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == 'scnet50':
        from net.SCNet.scnet import scnet50
        model = scnet50(pretrained=False)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    else:
        return

    model.input_space = 'RGB'
    model.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    model.input_range = [0, 1]
    model.mean = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5] for inception* networks
    model.std = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5] for inception* networks

    # 通过随机变化来进行数据增强
    test_tf = transforms.Compose([
        Preproc(0.2),
        # Rescale(IMAGE_SIZE),
        # transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪
        Resize(IMAGE_SIZE),  # 非等比例缩小
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
    ])

    test_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'val', test_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True
    )
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    #
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model = torch.load(model_path)

    model = model.cuda()

    criterion = nn.BCELoss(reduction='mean')  # 多分类用BCELoss

    avg, loss = test(model, test_loader, criterion)
    print('avg:', avg, 'loss:', loss)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print('总耗时', end - start)
