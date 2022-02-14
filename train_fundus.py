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
# from torch.utils.tensorboard import SummaryWriter
from utils.utils import calc_kappa,hamming_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cols = ['黄斑区视网膜出血', '黄斑区视网膜渗出', '黄斑区玻璃膜疣', '视网膜下橘红色病灶', '视网膜下出血']
classCount = len(cols)
data_dir = 'AMD_processed/'
list_dir = 'AMD_processed/label/new_two_stream/fundus/'
pre_models = \
        {"resnet18": models.resnet18,
         "resnet34": models.resnet34,
         "resnet50": models.resnet50,
         "resnest50": resnest50,
         "scnet50": scnet50,
         "inceptionv3": models.inception_v3,
         "vgg16": models.vgg16,
         "vgg19": models.vgg19}

mean = {
    224 : [0.485, 0.456, 0.406],
    299 : [0.5, 0.5, 0.5]
}
std = {
    224 : [0.229, 0.224, 0.225],
    299 : [0.5, 0.5, 0.5]
}

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

    parser.add_argument('--epoch', type=int, default=500, help = 'The number of training epoch')
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

def pred2int(x):
    out = []
    for i in range(len(x)):
        # print(x[i])
        out.append([1 if y > 0.5 else 0 for y in x[i]])
    return out

def train(model, train_loader, optimizer, scheduler, criterion, epoch, log):
    print(f'Epoch={epoch}\n')
    log.write(f'Epoch={epoch}\n')
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
    kappa = calc_kappa(y_true, y_pred, cols)
    y_pred = pred2int(y_pred)
    # print(y_pred)
    # print(y_true)
    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true, y_pred)
    avg = (f1 + precision + recall + auroc) / 4.0
    # writer.add_scalar("Train/f1", f1, epoch)
    # writer.add_scalar("Train/auroc", auroc, epoch)f
    # writer.add_scalar("Train/recall", recall, epoch)
    # writer.add_scalar("Train/precision", precision, epoch)
    # writer.add_scalar("Train/acc", acc, epoch)
    # writer.add_scalar("Train/avg", avg, epoch)
    # writer.add_scalar("Train/hamming_loss", hamming, epoch)
    # writer.add_scalar("Train/ELoss", out_loss, epoch)
    tbar.close()
    print()
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.format('f1', 'auroc', 'recall', 'precision', 'acc', 'kappa', 'hamming', 'loss'))
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)), str(round(acc,4)), str(round(kappa,4)), str(round(hamming,4)), str(round(out_loss,4)) ))
# print(f1, auroc, recall, precision, acc, avg, hamming)


def validate(model, val_loader, criterion, epoch, log):
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
    kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = pred2int(y_pred)
    # sw = compute_sample_weight(class_weight='balanced', y=y_true)

    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

    avg = (f1 + recall + precision + auroc) / 4.0
    # writer.add_scalar("Val/f1", f1, epoch)
    # writer.add_scalar("Val/auroc", auroc, epoch)
    # writer.add_scalar("Val/recall", recall, epoch)
    # writer.add_scalar("Val/precision", precision, epoch)
    # writer.add_scalar("Val/acc", acc, epoch)
    # writer.add_scalar("Val/avg", avg, epoch)
    # writer.add_scalar("Val/hamming_loss", hamming, epoch)
    # writer.add_scalar("Val/ELoss", out_loss, epoch)
    tbar.close()
    print()
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.format('f1', 'auroc', 'recall', 'precision', 'acc', 'kappa', 'hamming', 'loss'))
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)), str(round(acc,4)), str(round(kappa,4)), str(round(hamming,4)), str(round(out_loss,4)) ))

    # print(f1, auroc, recall, precision, acc, avg, hamming)
    return avg


def  main():
    model = pre_models[args.fundus_model](pretrained=True)
    if 'vgg' in args.fundus_model:
        model.classifier[3] = nn.Linear(in_features=4096, out_features=1000, bias=True)
        model.classifier[-1] = nn.Linear(in_features=1000, out_features=classCount, bias=True)
        model.classifier.add_module(f'{len(model.classifier)}',nn.Sigmoid())
    else:
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())


    train_tf = transforms.Compose([
        Resize(args.fundus_size),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        Resize(args.fundus_size),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'train', train_tf, classCount, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'val', val_tf, classCount, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    model = model.cuda()

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
    train_log = open('logs/fundus/'+ model_name + '-train.log', 'w')
    val_log = open('logs/fundus/'+ model_name + '-val.log', 'w')
    max_avg = 0

    for epoch in range(0, args.epoch):
        train(model, train_loader, optimizer, scheduler, criterion, epoch, train_log)
        avg = validate(model, val_loader, criterion, epoch, val_log)
        if avg > max_avg:
            torch.save(model, './model/fundus/' + model_name + '.pth')
            max_avg = avg



if __name__ == '__main__':
    args = get_parser()
    NAME = str(args.epoch) + "+" + str(args.learning_rate) + '+' + str(args.weight_decay) + '+' + args.loss
    model_name = datetime.datetime.now().strftime('%Y-%m-%d') + '+' + args.fundus_model + '+' + NAME
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
    data_dir = os.path.join(args.root_path, data_dir)
    list_dir = os.path.join(args.root_path, list_dir)
    # writer = SummaryWriter(os.path.join('runs', 'fundus/' + model_name[:-4]))
    print("Train fundus ", model_name)
    start = time.time()
    main()
    end = time.time()
    print('Finish fundus, Time=', end - start)
