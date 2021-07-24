import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize, ImgTrans
from data.csv_dataset import TwoStreamDataset
from utils.utils import calc_kappa
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

import time
import torch.nn.functional as F
from net.two_stream import TwoStreamNet, Only_OCT_Net
import albumentations
import cv2

from utils.Message import message

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

METHOD = ''
FUNDUS_MODEL = "scnet50"
OCT_MODEL = 'scnet50'

START_EPOCH = 0
EPOCHS = 100
BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1
LOSS = 'bceloss'

AVERAGE = 'weighted'

MOMENTUM = 0.9  # RECEIVED_PARAMS["momentum"]
WEIGHT_DECAY = 0.001  # RECEIVED_PARAMS["weight_decay"]
LR = 0.001  # RECEIVED_PARAMS["learning_rate"]

FUNDUS_IMAGE_SIZE = 224
OCT_IMAGE_SIZE = 224  # RECEIVED_PARAMS["image_size"]

cols = ['新生血管性AMD', 'PCV', '其他']
classCount = len(cols)

RESUME = False
NAME = METHOD + "+" + str(EPOCHS) + "+" + str(LR) + '+' + str(WEIGHT_DECAY) + '+' + LOSS
OCT_path = './model/OCT/2021_07_23+scnet50++500+0.001+0.0001+bceloss.pth'
model_name = '2021_07_24+' + FUNDUS_MODEL + '+' + OCT_MODEL + '+' + NAME + '.pth'

print("Train only fundus ", model_name, 'RESUME:', RESUME)

data_dir = '/home/hutianyi/datasets/AMD_processed/'
list_dir = '/home/hutianyi/datasets/AMD_processed/label/new_two_stream/'


def train(model, train_loader, optimizer, scheduler, criterion, writer, epoch):
    model.train()

    tbar = tqdm(train_loader, desc='\r', ncols=100)  # 进度条
    y_pred = []
    y_true = []
    loss_val = 0
    loss_val_norm = 0
    for batch_idx, (fundus, OCT, target) in enumerate(tbar):
        fundus, OCT, target = fundus.cuda(), OCT.cuda(), target.cuda()  # fundus.cuda(),target.cuda()
        optimizer.zero_grad()
        output = model(fundus, OCT)

        loss = criterion(output, target)
        loss.backward()
        #optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面
        optimizer.step()

        output_real = torch.argmax(F.softmax(output.cpu(), dim=1), dim=1)  # 单分类用softmax
        output_one_hot = F.one_hot(output_real, classCount)
        target_one_hot = F.one_hot(target, classCount)
        y_pred.extend(output_one_hot.numpy())
        y_true.extend(target_one_hot.data.cpu().numpy())

        writer.add_scalar("Train/loss", loss.item(), epoch * len(train_loader) + batch_idx)
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(OCT), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        #         print(output.data.cpu().numpy())
        #         print(target.data.cpu().numpy())
        #         exit()
        loss_val += loss.item()
        loss_val_norm += 1

    #todo(hty):对model1、model2及最后fc中所有的参数都进行更新（会不会存在只更新最后的fc效果更好的可能性）？
    scheduler.step()
    out_loss = loss_val / loss_val_norm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)

    y_pred = (y_pred > 0.5)
    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    writer.add_scalar("Train/f1", f1, epoch)
    writer.add_scalar("Train/kappa", kappa, epoch)
    writer.add_scalar("Train/auroc", auroc, epoch)
    writer.add_scalar("Train/avg", avg, epoch)
    writer.add_scalar("Train/precision", precision, epoch)
    writer.add_scalar("Train/recall", recall, epoch)
    writer.add_scalar("Train/acc", acc, epoch)
    writer.add_scalar("Train/ELoss", out_loss, epoch)
    tbar.close()
    print(f1, kappa, auroc, recall, precision, acc, avg)


def validate(model, val_loader, criterion, writer, epoch):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)  # 进度条
    loss_val = 0
    loss_val_norm = 0
    print(tbar)
    for batch_idx, (fundus, OCT, target) in enumerate(tbar):
        fundus, OCT, target = fundus.cuda(), OCT.cuda(), target.cuda()  # fundus.cuda(),target.cuda()
        # optimizer.zero_grad()
        output = model(fundus, OCT)

        loss = criterion(output, target)

        output_real = torch.argmax(F.softmax(output.cpu(), dim=1), dim=1)  # 单分类用softmax
        output_one_hot = F.one_hot(output_real, classCount)
        target_one_hot = F.one_hot(target, classCount)
        y_pred.extend(output_one_hot.numpy())
        y_true.extend(target_one_hot.data.cpu().numpy())

        writer.add_scalar("Val/loss", loss.item(), epoch * len(val_loader) + batch_idx)
        tbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(OCT), len(val_loader.dataset),
            100. * batch_idx / len(val_loader), loss.item()))
        loss_val += loss.item()
        loss_val_norm += 1

    out_loss = loss_val / loss_val_norm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)

    y_pred = (y_pred > 0.5)
    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    writer.add_scalar("Val/f1", f1, epoch)
    writer.add_scalar("Val/kappa", kappa, epoch)
    writer.add_scalar("Val/auroc", auroc, epoch)
    writer.add_scalar("Val/avg", avg, epoch)
    writer.add_scalar("Val/precision", precision, epoch)
    writer.add_scalar("Val/recall", recall, epoch)
    writer.add_scalar("Val/acc", acc, epoch)
    writer.add_scalar("Val/ELoss", out_loss, epoch)
    print(f1, kappa, auroc, recall, precision, acc, avg)
    if epoch % 10 == 0:
        message('Train_Only_OCT_Epoch' + str(epoch), 'f1='+str(f1)+'\nauroc='+ str(auroc)+'\nrecall='+ str(recall)+'\nprecision='+ str(precision)+'\nacc='+ str(acc)+'\navg='+ str(avg)+'\nhamming=')
    tbar.close()
    return avg


def main():
    model = Only_OCT_Net(OCT_path=OCT_path, fundus_model=FUNDUS_MODEL, OCT_model=OCT_MODEL,
                         num_classes=classCount)

    # model.input_space = 'RGB'
    # model.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    # model.input_range = [0, 1]
    # model.mean = [0.485, 0.456, 0.406]
    # model.std = [0.229, 0.224, 0.225]

    train_OCT_tf = transforms.Compose([
        Preproc(0.2),
        Resize(OCT_IMAGE_SIZE),  # 非等比例缩小
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ])
    val_OCT_tf = transforms.Compose([
        Preproc(0.2),
        Resize(OCT_IMAGE_SIZE),  # 非等比例缩小
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
    ])

    train_fundus_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(FUNDUS_IMAGE_SIZE),  # 等比例缩小
        transforms.CenterCrop(FUNDUS_IMAGE_SIZE),  # 以中心裁剪，fundus适用，OCT不适用
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
    ])
    val_fundus_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(FUNDUS_IMAGE_SIZE),  # 等比例缩小
        transforms.CenterCrop(FUNDUS_IMAGE_SIZE),  # 以中心裁剪
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
    ])

    train_loader = torch.utils.data.DataLoader(
        TwoStreamDataset(data_dir, 'train', train_fundus_tf, train_OCT_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        TwoStreamDataset(data_dir, 'val', val_fundus_tf, val_OCT_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )

    # if RESUME:
    #     model = torch.load(model_path)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)

    max_avg = 0

    writer = SummaryWriter(os.path.join('runs', 'only_OCT_' + model_name[:-4]))
    for epoch in range(START_EPOCH, EPOCHS):
        train(model, train_loader, optimizer, scheduler, criterion, writer, epoch)
        avg = validate(model, val_loader, criterion, writer, epoch)

        if avg > max_avg:
            torch.save(model, './model/only_OCT/' + model_name)
            max_avg = avg
    writer.close()

if __name__ == '__main__':
    import time
    message('开始训练Train_Only_OCT', '模型为'+model_name)
    start = time.time()
    main()
    end = time.time()
    message('完成训练Train_Only_OCT', '总耗时'+str(end - start))
    print('总耗时', end - start)
