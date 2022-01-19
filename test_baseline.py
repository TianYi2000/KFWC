import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from torchvision import transforms
from tqdm import tqdm

from data.base_dataset import Preproc, Rescale, ToTensor, Resize
from data.csv_dataset import TwoStreamDataset
from utils.utils import calc_kappa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

FUNDUS_IMAGE_SIZE = 224
OCT_IMAGE_SIZE = 224

AVERAGE = 'weighted'

cols = ['新生血管性AMD', 'PCV', '其他']
classCount = len(cols)
MODEL_PATHS = [
               # './model/baseline/2021_08_16+resnet18+resnet18++100+0.001+0.001+bceloss.pth',
               # './model/baseline/2021_08_16+resnet34+resnet34++100+0.001+0.001+bceloss.pth',
               # './model/baseline/2021_08_16+resnet50+resnet50++100+0.001+0.001+bceloss.pth',
               # './model/baseline/2021_08_16+inceptionv3+inceptionv3++100+0.001+0.001+bceloss.pth',
               # './model/baseline/2021_08_16+scnet50+scnet50++100+0.001+0.001+bceloss.pth',
               # './model/baseline/2021_08_16+resnest50+resnest50++100+0.001+0.001+bceloss.pth',
               #  './model/baseline/2021_09_08+resnet50+resnet50++100+0.001+0.001+bceloss.pth',
                './model/baseline/2021_10_29+resnet50+resnet50++100+0.001+0.001+bceloss.pth',
                './model/baseline/2021_10_29+scnet50+scnet50++100+0.001+0.001+bceloss.pth',
               ]


data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/'


def test(model, val_loader, criterion):
    model.eval()
    y_pred = []
    y_true = []
    y_pred_prob=[]
    tbar = tqdm(val_loader, desc='\r', ncols=100)  # 进度条
    loss_val = 0
    loss_val_norm = 0
    # print(tbar)
    for batch_idx, (fundus, OCT, target) in enumerate(tbar):
        fundus, OCT, target = fundus.cuda(), OCT.cuda(), target.cuda()  # fundus.cuda(),target.cuda()
        # optimizer.zero_grad()
        output = model(fundus, OCT)

        loss = criterion(output, target)

        output_faltten = F.softmax(output.cpu(), dim=1)
        y_pred_prob.extend(output_faltten.data.numpy())
        output_real = torch.argmax(output_faltten, dim=1)  # 单分类用softmax

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
    y_pred_prob = np.array(y_pred_prob)
    auroc = roc_auc_score(y_true, y_pred_prob, average=AVERAGE)

    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    print( auroc,',',precision,',', recall,',',f1,',', kappa, ' .... ',acc, avg)
    tbar.close()
    # print("AUROC\t=\t", auroc)
    return avg, out_loss


def main(model_path):

    if 'incep' in model_path:
        FUNDUS_IMAGE_SIZE = 299
        OCT_IMAGE_SIZE = 299

        test_OCT_tf = transforms.Compose([
            Preproc(0.2),
            Resize(OCT_IMAGE_SIZE),  # 非等比例缩小
            ToTensor(),
            transforms.Normalize( [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # resnet和inception不同
            #
        ])

        test_fundus_tf = transforms.Compose([
            Preproc(0.2),
            Rescale(FUNDUS_IMAGE_SIZE),  # 等比例缩小
            transforms.CenterCrop(FUNDUS_IMAGE_SIZE),  # 以中心裁剪
            ToTensor(),
            transforms.Normalize( [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # resnet和inception不同
        ])

    else:
        FUNDUS_IMAGE_SIZE = 224
        OCT_IMAGE_SIZE = 224

        test_OCT_tf = transforms.Compose([
            Preproc(0.2),
            Resize(OCT_IMAGE_SIZE),  # 非等比例缩小
            ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # resnet和inception不同
            #
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
    # print('avg:', avg, 'loss:', loss)


if __name__ == '__main__':
    import time

    start = time.time()

    for model_path in MODEL_PATHS:
        print("Test baseline ", model_path)
        main(model_path)
    end = time.time()
    print('总耗时', end - start)
