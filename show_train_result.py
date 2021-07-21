import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize
from data.csv_dataset import MultiDataset
from utils.utils import calc_kappa
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

import time
import csv
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

METHOD = 'first_OCT_data'
MODEL = "resnet34"
LOSS = 'bceloss'

START_EPOCH = 0
EPOCHS = 50
classCount = 12
NAME = METHOD + "+" + str(EPOCHS) + "+" + LOSS

model_path = './model/OCT/2020_09_28+scnet50+first_OCT_data+200+bceloss.pth'
model_name = '2020_10_06+' + MODEL + '+' + NAME + '.pth'
print("Train OCT:", model_path)

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

MOMENTUM = 0.9  # RECEIVED_PARAMS["momentum"]
WEIGHT_DECAY = 0.0001  # RECEIVED_PARAMS["weight_decay"]
LR = 0.01  # RECEIVED_PARAMS["learning_rate"]

IMAGE_SIZE = 224  # RECEIVED_PARAMS["image_size"]

AVERAGE = 'weighted'

cols = ['视网膜前膜', '视网膜层间高反射灶', '色素上皮脱离', '神经上皮脱离', '黄斑裂孔', '黄斑囊样水肿',
        '玻璃膜疣', '玻璃体皮质后脱离', '视网膜增厚', '中浆', '视网膜椭圆体带反光减弱', '视网膜分支静脉阻塞']

# 训练的df 路径
data_dir = '/home/chaiwenjun/dataset/xingtai/export-final/1594352689.0330381/images/'
list_dir = '/home/hejiawen/datasets/xingtai_AMD/label/multi_OCT/'


def train(model, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()

    tbar = tqdm(train_loader, desc='\r', ncols=100)  # 进度条
    y_pred = []
    y_true = []
    lossVal = 0
    lossValNorm = 0
    print(tbar)
    for batch_idx, (inputs, target) in enumerate(tbar):
        target = target.float()
        data, target = inputs.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        # start magic loss
        if LOSS == 'magicloss':
            no_of_classes = classCount
            beta = 0.9999

            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * no_of_classes

            labels_one_hot = target
            # labels_one_hot =target
            weights = torch.tensor(weights).float()
            weights = weights.unsqueeze(0)
            weights = weights.repeat(labels_one_hot.shape[0], 1).cuda() * labels_one_hot.cuda()

            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, no_of_classes)
            pred = output  # .softmax(dim = 1)
            loss = F.binary_cross_entropy(pred, labels_one_hot, weight=weights)
        # end magic loss
        else:
            # loss_1 = criterion[0](output, target)
            # loss_2 = criterion[1](output, target)
            # loss = loss_1 + loss_2
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

        lossVal += loss.item()
        lossValNorm += 1

    scheduler.step()
    outLoss = lossVal / lossValNorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if epoch >= 49:
        fw = open('logs/train_result.csv', 'w', encoding='utf-8', newline='')
        writer = csv.writer(fw)
        title = ['ground truth', 'prediction']
        writer.writerow(title)

        for idx in range(len(y_pred)):
            data = list()
            data.append(y_true[idx])

            y_idx = list()
            for y in y_pred[idx]:
                y_idx.append('{:.5f}'.format(y))
            data.append(y_idx)

            writer.writerow(data)

        fw.close()

    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)

    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + recall + auroc) / 4.0
    tbar.close()
    print(auroc, f1, kappa, avg, precision, recall, acc)


def validate(model, val_loader, criterion, epoch):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)
    lossVal = 0
    lossValNorm = 0
    for batch_idx, (inputs, target) in enumerate(tbar):
        target = target.float()  # 多分类用.float()
        data, target = inputs.cuda(), target.cuda()  # leftImg.cuda(),target.cuda()

        output = model(data)

        # start magic loss
        if LOSS == 'magicloss':
            no_of_classes = classCount

            beta = 0.9999

            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * no_of_classes

            labels_one_hot = target
            weights = torch.tensor(weights).float()
            weights = weights.unsqueeze(0)
            weights = weights.repeat(labels_one_hot.shape[0], 1).cuda() * labels_one_hot.cuda()

            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, no_of_classes)
            pred = output  # .softmax(dim = 1)
            loss = F.binary_cross_entropy(pred, labels_one_hot, weight=weights)
        # end magic loss
        else:
            # loss_1 = criterion[0](output, target)
            # loss_2 = criterion[1](output, target)
            # loss = loss_1 + loss_2
            loss = criterion(output, target)

        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        tbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(val_loader.dataset),
                   100. * batch_idx / len(val_loader), loss.item()))
        lossVal += loss.item()
        lossValNorm += 1

    outLoss = lossVal / lossValNorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if epoch >= 49:
        fw = open('logs/val_result.csv', 'w', encoding='utf-8', newline='')
        writer = csv.writer(fw)
        title = ['ground truth', 'prediction']
        writer.writerow(title)

        for idx in range(len(y_pred)):
            data = list()
            data.append(y_true[idx])

            y_idx = list()
            for y in y_pred[idx]:
                y_idx.append('{:.5f}'.format(y))
            data.append(y_idx)

            writer.writerow(data)

        fw.close()

    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)

    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + recall + kappa + auroc) / 4.0
    tbar.close()
    print(auroc, f1, kappa, avg, precision, recall, acc)
    return avg


def main():
    if MODEL == "resnet34":
        model = models.resnet34(pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())  # 多分类用sigmoid
    elif MODEL == "resnet50":
        model = models.resnet50(pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    elif MODEL == "resnest50":
        from resnest.torch import resnest50
        model = resnest50(pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    elif MODEL == "inceptionv3":
        model = models.inception_v3(pretrained=False, aux_logits=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    elif MODEL == 'scnet50':
        from net.SCNet.scnet import scnet50
        model = scnet50(pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    else:
        return

    model.input_space = 'RGB'
    model.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    model.input_range = [0, 1]
    model.mean = [0.485, 0.456, 0.406]
    model.std = [0.229, 0.224, 0.225]

    # 通过随机变化来进行数据增强
    train_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        MultiDataset(data_dir, 'train', train_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        MultiDataset(data_dir, 'val', val_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )

    model = model.cuda()

    criterion = nn.BCELoss(size_average=True)  # 多分类用BCELoss
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
    maxAvg = 0

    for epoch in range(START_EPOCH, EPOCHS):
        train(model, train_loader, optimizer, scheduler, criterion, epoch)
        avg = validate(model, val_loader, criterion, epoch)

        if avg > maxAvg:
            torch.save(model, './model/OCT/' + model_name)
            maxAvg = avg


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print('总耗时', end - start)

