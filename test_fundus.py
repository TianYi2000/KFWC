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
from utils.utils import Cal_Threshold
# from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cols = ['Intraretinal fluid',
        'Subretinal fluid',
        'Pigment epithelial',
        'Hyperreflective lesions under RPE',
        'Hyperreflective lesions in or under the retina']
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
    parser.add_argument('--fundus_path', type=str, help='the model file path of fundus model')
    parser.add_argument('--oct_path', type=str, help='the model file path of OCT model')
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

def pred2int(x, thresholds):
    out = []
    for i in range(len(x)):
        # print(x[i])
        out.append([1 if y > thresholds[j] else 0 for j, y in enumerate(x[i])] )
    return np.array(out)


def test(model, val_loader, criterion):
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
            tbar.set_description('Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(inputs), len(val_loader.dataset),
                100. * batch_idx / len(val_loader), loss.item()))
            loss_val += loss.item()
            loss_valnorm += 1

    out_loss = loss_val / loss_valnorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=args.average)
    # kappa = calc_kappa(y_true, y_pred, cols)
    thresholds = Cal_Threshold(y_true, y_pred)
    y_pred = pred2int(y_pred, thresholds)
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
    print('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('f1', 'auroc', 'recall', 'precision', 'acc', 'avg', 'hamming', 'loss'))
    print('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)), str(round(acc,4)), str(round(avg,4)), str(round(hamming,4)), str(round(out_loss,4)) ))

    # print(f1, auroc, recall, precision, acc, avg, hamming)
    return avg


def  main():
    model = torch.load(args.fundus_path)

    test_tf = transforms.Compose([
        Resize(args.fundus_size),
        ToTensor(),
        transforms.Normalize(mean=mean[args.fundus_size], std=std[args.fundus_size])
    ])

    test_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'test', test_tf, classCount, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    model = model.cuda()

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    test(model, test_loader, criterion)



if __name__ == '__main__':
    args = get_parser()
    NAME = str(args.epoch) + "+" + str(args.learning_rate) + '+' + str(args.weight_decay) + '+' + args.loss
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
    data_dir = os.path.join(args.root_path, data_dir)
    list_dir = os.path.join(args.root_path, list_dir)
    print(f"Test fundus {args.fundus_path}")
    start = time.time()
    main()
    end = time.time()
    print('Finish fundus, Time=', end - start)
