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
from utils.draw import draw_roc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATHS=[
    # './model/two_stream/2021_08_16+resnet18+resnet18++100+0.001+0.001+bceloss.pth',
    #          './model/two_stream/2021_08_16+resnet34+resnet34++100+0.001+0.001+bceloss.pth',
    #          './model/two_stream/2021_05_21+resnet34+resnet34++100+0.001+0.001+bceloss.pth'
             './model/baseline/2021_08_16+resnest50+resnest50++100+0.001+0.001+bceloss.pth',
            # './model/two_stream/2021_05_26+inceptionv3+inceptionv3++100+0.001+0.001+bceloss.pth',
            #  './model/two_stream/2021_08_10+scnet50+scnet50++100+0.001+0.001+bceloss.pth',
             './model/two_stream/2021_08_12+resnest50+resnest50++100+0.001+0.001+bceloss.pth',

             # './model/two_stream/2021_05_26+scnet50+scnet50++100+0.001+0.001+bceloss.pth',
             # './model/two_stream/2021_08_10+scnet50+scnet50++100+0.001+0.001+bceloss.pth',
]

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1
LOSS = 'bceloss'

FUNDUS_IMAGE_SIZE = 224
OCT_IMAGE_SIZE = 224

AVERAGE = 'weighted'

cols = ['新生血管性AMD', 'PCV', '其他']
classCount = len(cols)

data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/'


def test(model, test_loader, criterion):
    model.eval()
    y_pred = []
    y_true = []
    y_pred_prob = []
    tbar = tqdm(test_loader, desc='\r', ncols=100)  # 进度条
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
            batch_idx * len(OCT), len(test_loader.dataset),
            100. * batch_idx / len(test_loader), loss.item()))
        loss_val += loss.item()
        loss_val_norm += 1


    out_loss = loss_val / loss_val_norm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    auroc = roc_auc_score(y_true, y_pred_prob, average=AVERAGE)
    # draw_roc(auroc, y_pred_prob, y_true )
    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    print( auroc,',',precision,',', recall,',',f1,',', kappa, ' .... ',acc, avg)
    # print("AUROC\t=\t", auroc)
    tbar.close()
    return avg, out_loss


def main(model_path):
    # model.input_space = 'RGB'
    # model.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    # model.input_range = [0, 1]
    # model.mean = [0.485, 0.456, 0.406]
    # model.std = [0.229, 0.224, 0.225]

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

    # if RESUME:
    #     model = torch.load(model_path)

    model = torch.load(model_path)
    # print(model)
    # return
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    avg, loss = test(model, test_loader, criterion)
    # print('avg:', avg, 'loss:', loss)


if __name__ == '__main__':
    import time

    start = time.time()
    for model_path in MODEL_PATHS:
        print("Test two_stream:", model_path)
        main(model_path)
    end = time.time()
    print('总耗时', end - start)
