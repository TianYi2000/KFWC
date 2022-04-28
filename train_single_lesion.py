import os
import time
import datetime
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize, ImgTrans
from data.csv_dataset import Lesion_Complaint_Dataset
from utils.utils import calc_kappa
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, \
    hamming_loss
import time
import torch.nn.functional as F
from net.three_stream import Single_Complaint_Net
import cv2
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cols = ['新生血管性AMD', 'PCV', '其他']
lesion_text = ['视网膜内液性暗腔','视网膜下积液','RPE脱离','RPE下高反射病灶','视网膜内或视网膜下高反射病灶','尖锐的RPED峰','双层征','多发性RPED','RPED切迹','视网膜内高反射硬性渗出']
classCount = len(cols)
lesion_num = len(lesion_text)
data_dir = 'AMD_processed/'
list_dir = '主诉/saved-OCT图像-疾病-体征-主诉-重新配对主诉/'

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
    parser.add_argument('--root_path', type=str, default='.',
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
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')

    parser.add_argument('--workers', type=int, default=1, help='The number of sub-processes to use for data loading')
    parser.add_argument('--average', type=str, default='weighted',
                        choices=['micro', 'macro', 'weighted', 'samples'],
                        help='the type of averaging performed on the data')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum in optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight_decay in optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning_rate in optimizer')
    parser.add_argument('--loss', type=str, default='bceloss', help='The loss function')
    parser.add_argument('--text_model', type=str,default='bert', help='text model')
    parser.add_argument('--lock_text_weight', action='store_true', help='lock_text_weight')
    parser.add_argument('--use_gpu', type=str, default='2,3', help='The GPU on server used')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')

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
    tbar = tqdm(train_loader, desc='\r', ncols=100)  # 进度条
    y_pred = []
    y_true = []
    loss_val = 0
    loss_val_norm = 0
    for batch_idx, (image, lesion_id, lesion_mask, lesion_type, complaint_id, complaint_mask, complaint_type, target) in enumerate(tbar):
        image, target = image.cuda(), target.cuda()
        lesion_id, lesion_mask, lesion_type, complaint_id, complaint_mask, complaint_type = \
            lesion_id.cuda(), lesion_mask.cuda(), lesion_type.cuda(), complaint_id.cuda(), complaint_mask.cuda(), complaint_type.cuda()
        target = target.long()
        optimizer.zero_grad()
        output = model(image, lesion_id, lesion_mask, lesion_type)
        loss = criterion(output, target)
        loss.backward()
        #optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面
        optimizer.step()

        output_real = torch.argmax(output.cpu(), dim=1)
        output_one_hot = F.one_hot(output_real, classCount)
        target_one_hot = F.one_hot(target, classCount)
        y_pred.extend(output_one_hot.numpy())
        y_true.extend(target_one_hot.data.cpu().numpy())
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        #         print(output.data.cpu().numpy())
        #         print(target.data.cpu().numpy())
        #         exit()
        loss_val += loss.item()
        loss_val_norm += 1

    scheduler.step()
    out_loss = loss_val / loss_val_norm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=args.average)

    y_pred = pred2int(y_pred)
    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    print()
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.
              format('f1', 'auroc', 'recall', 'precision', 'acc', 'kappa', 'loss'))
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.
              format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)),
                     str(round(acc,4)), str(round(kappa,4)), str(round(out_loss,4)) ))


def val(model, val_loader, criterion, epoch, log):
    log.write(f'Epoch={epoch}\n')
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)  # 进度条
    loss_val = 0
    loss_val_norm = 0
    with torch.no_grad():
        for batch_idx, (image, lesion_id, lesion_mask, lesion_type, complaint_id, complaint_mask, complaint_type, target) in enumerate(tbar):
            image, target = image.cuda(), target.cuda()
            lesion_id, lesion_mask, lesion_type, complaint_id, complaint_mask, complaint_type = \
                lesion_id.cuda(), lesion_mask.cuda(), lesion_type.cuda(), complaint_id.cuda(), complaint_mask.cuda(), complaint_type.cuda()
            # target = torch.tensor(target, dtype=torch.long).clone().detach()
            target = target.long()
            output = model(image, lesion_id, lesion_mask, lesion_type)

            loss = criterion(output, target)

            output_real = torch.argmax(output.cpu(), dim=1)  # 单分类用softmax
            output_one_hot = F.one_hot(output_real, classCount)
            target_one_hot = F.one_hot(target, classCount)
            y_pred.extend(output_one_hot.numpy())
            y_true.extend(target_one_hot.data.cpu().numpy())

            tbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(val_loader.dataset),
                100. * batch_idx / len(val_loader), loss.item()))
            loss_val += loss.item()
            loss_val_norm += 1

    out_loss = loss_val / loss_val_norm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=args.average)

    y_pred = pred2int(y_pred)
    f1 = f1_score(y_true, y_pred, average=args.average)
    precision = precision_score(y_true, y_pred, average=args.average)
    recall = recall_score(y_true, y_pred, average=args.average)
    kappa = calc_kappa(y_true, y_pred, cols)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    avg = (f1 + kappa + auroc + recall) / 4.0
    print()
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.
              format('f1', 'auroc', 'recall', 'precision', 'acc', 'kappa', 'loss'))
    log.write('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n'.
              format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)),
                     str(round(acc,4)), str(round(kappa,4)), str(round(out_loss,4)) ))

    return avg


def main():
    model = Single_Complaint_Net(image_model=args.oct_model, lock_text_weight = args.lock_text_weight)


    train_tf = transforms.Compose([
        Resize(args.oct_size),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize(mean=mean[args.oct_size], std=std[args.oct_size])
    ])
    val_tf = transforms.Compose([
        Resize(args.oct_size),
        ToTensor(),
        transforms.Normalize(mean=mean[args.oct_size], std=std[args.oct_size])
    ])
    train_loader = torch.utils.data.DataLoader(
        Lesion_Complaint_Dataset(data_dir, 'train', train_tf, classCount, lesion_num, lesion_text, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        Lesion_Complaint_Dataset(data_dir, 'val', val_tf, classCount, lesion_num, lesion_text, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    # if RESUME:
    #     model = torch.load(model_path)
    if ',' in args.use_gpu:
        torch.distributed.init_process_group(backend="nccl")
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = model.cuda()


    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)


    train_log = open('logs/single_lesion/'+ model_name + '-train.log', 'w')
    val_log = open('logs/single_lesion/'+ model_name + '-val.log', 'w')
    max_avg = 0

    for epoch in range(0, args.epoch):
        train(model, train_loader, optimizer, scheduler, criterion, epoch, train_log)
        avg = val(model, val_loader, criterion, epoch, val_log)

        if avg > max_avg:
            torch.save(model, './model/single_lesion/' + model_name + '.pth')
            max_avg = avg


if __name__ == '__main__':
    args = get_parser()
    NAME = str(args.epoch) + "+" + str(args.learning_rate) + '+' + str(args.weight_decay) + '+' + args.loss
    model_name = datetime.datetime.now().strftime('%Y-%m-%d') + '+' + args.oct_model + '+' + args.text_model + ('_lock' if args.lock_text_weight else '') + '+'  + NAME
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu
    data_dir = os.path.join(args.root_path, data_dir)
    list_dir = os.path.join(args.root_path, list_dir)
    # writer = SummaryWriter(os.path.join('runs', 'OCT/' + model_name[:-4]))
    print("Train Single Lesion Net ", model_name)
    start = time.time()
    main()
    end = time.time()
    print('Finish Single Lesion Net, Time=', end - start)
