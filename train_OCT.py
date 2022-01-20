import os
import time
import datetime
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, models
from resnest.torch import resnest50
from net.SCNet.scnet import scnet50
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, \
    hamming_loss
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize, ImgTrans
from data.csv_dataset import ImageDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cols = ['Intraretinal fluid',
        'Subretinal fluid',
        'Pigment epithelial',
        'Hyperreflective lesions under RPE',
        'Hyperreflective lesions in or under the retina']
classCount = len(cols)
data_dir = 'AMD_processed/'
list_dir = 'AMD_processed/label/new_two_stream/OCT/'
samples_per_cls = [445, 543, 808, 109, 1090, 23, 241, 88, 21, 158]
pre_models = \
        {"resnet18": models.resnet18,
         "resnet34": models.resnet34,
         "resnet50": models.resnet50,
         "resnest50": resnest50,
         "scnet50": scnet50,
         "inceptionv3": models.inception_v3,
         "vgg16": models.vgg16,
         "vgg19": models.vgg19}

def get_parser():
    parser = argparse.ArgumentParser(description='Input hyperparameter of model:')
    parser.add_argument('--root_path', type=str, default='/home/hejiawen/datasets',
                            help='The root path of dataset')
    parser.add_argument('--fundus_model', type=str, default='resnet50',
                            choices=['resnet18', 'resnet34', 'resnet50', 'resnest50', 'scnet50', 'inceptionv3', 'vgg16', 'vgg19'],
                            help='The backbone model for Color fundus image')
    parser.add_argument('--oct_model', type=str, default='resnet50',
                            choices=['resnet18', 'resnet34', 'resnet50', 'resnest50', 'scnet50', 'inceptionv3', 'vgg16', 'vgg19'],
                            help='The backbone model for OCT image')

    parser.add_argument('--fundus_size', type=int, default=224, help='The input size for Color fundus image')
    parser.add_argument('--oct_size', type=int, default=224, help='The input size for OCT image')

    parser.add_argument('--epoch', type=int, default=100, help = 'The number of training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')

    parser.add_argument('--workers', type=int, default=1, help='The number of sub-processes to use for data loading')
    parser.add_argument('--average', type=str, default='weighted',
                        choices=['micro', 'macro', 'weighted', 'samples'],
                        help='the type of averaging performed on the data')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum in optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='The weight_decay in optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning_rate in optimizer')
    parser.add_argument('--loss', type=str, default='bceloss', help='The loss function')

    parser.add_argument('--use_gpu', type=int, default=0, choices=[0,1,2,3], help='The GPU on server used', required=True)

    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    tbar = tqdm(train_loader, desc='\r', ncols=100)
    y_pred = []
    y_true = []
    loss_val = 0
    loss_valnorm = 0
    # print(tbar)
    for batch_idx, (inputs, target) in enumerate(tbar):
        target = target.float()
        data, target = inputs.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

        loss_val += loss.item()
        loss_valnorm += 1

    scheduler.step()
    out_loss = loss_val / loss_valnorm
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=args.average)
    y_pred = (y_pred > 0.5)

    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true, y_pred)
    avg = (f1 + precision + recall + auroc) / 4.0
    tbar.close()
    print(f1, auroc, recall, precision, acc, avg, hamming)


def validate(model, val_loader, criterion, epoch):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)
    loss_val = 0
    loss_valnorm = 0
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(tbar):
            target = target.float()
            data, target = inputs.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            y_pred.extend(output.data.cpu().numpy())
            y_true.extend(target.data.cpu().numpy())
            # writer.add_scalar("Val/loss", loss.item(), epoch * len(val_loader) + batch_idx)
            tbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(val_loader.dataset),
                100. * batch_idx / len(val_loader), loss.item()))
            loss_val += loss.item()
            loss_valnorm += 1

    out_loss = loss_val / loss_valnorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=args.average)
    # kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)
    # sw = compute_sample_weight(class_weight='balanced', y=y_true)

    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

    avg = (f1 + recall + precision + auroc) / 4.0
    tbar.close()
    print(f1, auroc, recall, precision, acc, avg, hamming)
    return avg


def  main():
    model = pre_models[args.oct_model](pretrained=True)
    kernel_count = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.oct_size),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        Resize(args.oct_size),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'train', train_tf, classCount, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'val', val_tf, classCount, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    model = model.cuda()

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)

    max_avg = 0

    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, scheduler, criterion, epoch)
        avg = validate(model, val_loader, criterion, epoch)
        if avg > max_avg:
            torch.save(model, './model/OCT/' + model_name)
            max_avg = avg



if __name__ == '__main__':
    args = get_parser()
    NAME = str(args.epoch) + "+" + str(args.learning_rate) + '+' + str(args.weight_decay) + '+' + args.loss
    model_name = datetime.datetime.now().strftime('%Y-%m-%d') + '+' + args.oct_model + '+' + NAME + '.pth'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
    data_dir = os.path.join(args.root_path, data_dir)
    list_dir = os.path.join(args.root_path, list_dir)
    print("Train OCT ", model_name)
    start = time.time()
    main()
    end = time.time()
    print('Finish OCT, Time=', end - start)
