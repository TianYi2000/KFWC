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

from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

import time
import torch.nn.functional as F
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

AVERAGE = 'weighted'

FUNDUS_IMAGE_SIZE = 224
OCT_IMAGE_SIZE = 224  # RECEIVED_PARAMS["image_size"]

cols = ['新生血管性AMD', 'PCV', '其他']
classCount = len(cols)
model_path = './model/baseline/2021_05_29+scnet50+scnet50++100+0.001+0.001+bceloss.pth'

print("Test baseline ", model_path)

data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/'


def test(model, val_loader, criterion):
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

        tbar.set_description('Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(OCT), len(val_loader.dataset),
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
    print(f1, kappa, auroc, recall, precision, acc, avg)
    tbar.close()
    return avg, out_loss


def main():
    test_OCT_tf = transforms.Compose([
        Preproc(0.2),
        Resize(OCT_IMAGE_SIZE),  # 非等比例缩小
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
        # [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ])

    test_fundus_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(FUNDUS_IMAGE_SIZE),  # 等比例缩小
        transforms.CenterCrop(FUNDUS_IMAGE_SIZE),  # 以中心裁剪
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
    ])

    test_loader = torch.utils.data.DataLoader(
        TwoStreamDataset(data_dir, 'test', test_fundus_tf, test_OCT_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )

    model = torch.load(model_path)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    avg, loss = test(model, test_loader, criterion)
    print('avg:', avg, 'loss:', loss)


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    end = time.time()
    print('总耗时', end - start)
