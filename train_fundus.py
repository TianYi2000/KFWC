import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize
from data.csv_dataset import ImageDataset
# from utils.utils import calc_kappa
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, hamming_loss
# from sklearn.utils.class_weight import compute_sample_weight

import time
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

METHOD = ''
LABEL = 'multilabel'
MODEL = "inceptionv3"
LOSS = 'bceloss'

START_EPOCH = 0
EPOCHS = 500

MOMENTUM = 0.9  # RECEIVED_PARAMS["momentum"]
WEIGHT_DECAY = 0.0001  # RECEIVED_PARAMS["weight_decay"]
LR = 0.001  # RECEIVED_PARAMS["learning_rate"]
NAME = METHOD + "+" + str(EPOCHS) + "+" + str(LR) + '+' + str(WEIGHT_DECAY) + '+' + LOSS

RESUME = False
model_path = ''
model_name = '2021_05_23+' + MODEL + '+' + NAME + '.pth'
# model_name = 'try_loss.pth'
print("Train Fundus:", model_name, 'RESUME:', RESUME)

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

IMAGE_SIZE = 299  # 224 for resnet, 299 for inception

AVERAGE = 'weighted'

cols = ['黄斑区视网膜出血', '黄斑区视网膜渗出', '黄斑区玻璃膜疣', '视网膜下橘红色病灶', '视网膜下出血']
classCount = len(cols)
samples_per_cls = [164, 220, 160, 26, 68]

# 训练的df 路径
data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/fundus/'


def train(model, train_loader, optimizer, scheduler, criterion, writer, epoch):
    model.train()

    tbar = tqdm(train_loader, desc='\r', ncols=100)  # 进度条
    y_pred = []
    y_true = []
    loss_val = 0
    loss_valnorm = 0
    print(tbar)
    for batch_idx, (inputs, target) in enumerate(tbar):
        target = target.float()
        data, target = inputs.cuda(), target.cuda()
        optimizer.zero_grad()
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
            # loss_1 = criterion[0](output, target)
            # loss_2 = criterion[1](output, target)
            # loss = loss_1 + loss_2
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        # writer.add_scalar("Train/loss", loss.item(), epoch * len(train_loader) + batch_idx)
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        #         print(output.data.cpu().numpy())
        #         print(target.data.cpu().numpy())
        #         exit()

        loss_val += loss.item()
        loss_valnorm += 1

    scheduler.step()
    out_loss = loss_val / loss_valnorm

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)
    # kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)
    # # 添加sample_weight试试
    # sw = compute_sample_weight(class_weight='balanced', y=y_true)

    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true, y_pred)

    avg = (f1 + precision + recall + auroc) / 4.0
    writer.add_scalar("Train/f1", f1, epoch)
    writer.add_scalar("Train/auroc", auroc, epoch)
    writer.add_scalar("Train/recall", recall, epoch)
    writer.add_scalar("Train/precision", precision, epoch)
    writer.add_scalar("Train/acc", acc, epoch)
    writer.add_scalar("Train/avg", avg, epoch)
    writer.add_scalar("Train/hamming_loss", hamming, epoch)
    writer.add_scalar("Train/ELoss", out_loss, epoch)
    tbar.close()
    print(f1, auroc, recall, precision, acc, avg, hamming)


def validate(model, val_loader, criterion, writer, epoch):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)
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
            # loss_1 = criterion[0](output, target)
            # loss_2 = criterion[1](output, target)
            # loss = loss_1 + loss_2
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
    auroc = roc_auc_score(y_true, y_pred, average=AVERAGE)
    # kappa = calc_kappa(y_true, y_pred, cols)

    y_pred = (y_pred > 0.5)
    # # 添加sample_weight试试
    # sw = compute_sample_weight(class_weight='balanced', y=y_true)

    f1 = f1_score(y_true, y_pred, average=AVERAGE)
    precision = precision_score(y_true, y_pred, average=AVERAGE)
    recall = recall_score(y_true, y_pred, average=AVERAGE)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

    avg = (f1 + recall + precision + auroc) / 4.0
    writer.add_scalar("Val/f1", f1, epoch)
    writer.add_scalar("Val/auroc", auroc, epoch)
    writer.add_scalar("Val/recall", recall, epoch)
    writer.add_scalar("Val/precision", precision, epoch)
    writer.add_scalar("Val/acc", acc, epoch)
    writer.add_scalar("Val/avg", avg, epoch)
    writer.add_scalar("Val/hamming_loss", hamming, epoch)
    writer.add_scalar("Val/ELoss", out_loss, epoch)
    tbar.close()
    print(f1, auroc, recall, precision, acc, avg, hamming)
    return avg


def main():
    if MODEL == "resnet18":
        model = models.resnet18(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())  # 多分类用sigmoid
    elif MODEL == "resnet34":
        model = models.resnet34(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "resnet50":
        model = models.resnet50(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "resnest50":
        from resnest.torch import resnest50
        model = resnest50(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    elif MODEL == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        kernel_count = model.AuxLogits.fc.in_features
        # model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
        model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
        model.aux_logits = False
    elif MODEL == 'scnet50':
        from net.SCNet.scnet import scnet50
        model = scnet50(pretrained=True)
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, classCount), nn.Sigmoid())
    else:
        return
    # ######################以下部分待修改#########################################
    """
    elif MODEL == "resnet152":
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=8, bias=True)
    elif MODEL == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    elif MODEL == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
        print(model)
        # model.classifier = torch.nn.Linear(in_features=36864, out_features=2, bias=True)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=8, bias=True)
        # model = torchvision_models.densenet121(num_classes=2, pretrained='imagenet')
    elif MODEL == "densenet161":
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=2208, out_features=8, bias=True)
        print(model)
    elif MODEL == "fbresnet152":
        model = pretrainedmodels.fbresnet152()
        model.last_linear = torch.nn.Linear(in_features=2048, out_features=8, bias=True)
        print(model)
        # model = fbresnet.fbresnet152(num_classes=2, pretrained='imagenet')
    elif MODEL == "inceptionv4":
        model = pretrainedmodels.inceptionv4()
        model.last_linear = torch.nn.Linear(in_features=1536, out_features=8, bias=True)
        print(model)
        # model = inceptionv4.inceptionv4(num_classes=2,pretrained='imagenet')
    elif MODEL == "se_resnext101":
        model = pretrainedmodels.se_resnext101_32x4d()
        print(model)
        model.last_linear = torch.nn.Linear(in_features=2048, out_features=8, bias=True)
    elif MODEL == "vgg19":
        model = pretrainedmodels.vgg19_bn()
        print(model)
        model.last_linear = torch.nn.Linear(in_features=4096, out_features=8, bias=True)
    elif MODEL == "vgg16":
        model = pretrainedmodels.vgg16_bn()
        print(model)
        model.last_linear = torch.nn.Linear(in_features=4096, out_features=8, bias=True)
    """
    if RESUME:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # lt = 8
    # cntr = 0
    # for child in model.children():
    #     cntr += 1
    #     if cntr < lt:
    #         print(child)
    #         for param in child.parameters():
    #             param.requires_grad = False

    model.input_space = 'RGB'
    model.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    model.input_range = [0, 1]
    model.mean = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406] for inception* networks
    model.std = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225] for resnet* networks

    # 通过随机变化来进行数据增强
    train_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(IMAGE_SIZE),    # 等比例缩小
        transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪，fundus适用，OCT不适用
        # Resize(IMAGE_SIZE),  # 非等比例缩小
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # resnet和inception不同
    ])
    val_tf = transforms.Compose([
        Preproc(0.2),
        Rescale(IMAGE_SIZE),    # 等比例缩小
        transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪
        # Resize(IMAGE_SIZE),     # 非等比例缩小
        ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # resnet和inception不同
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'train', train_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir, 'val', val_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    # model = nn.DataParallel(model).cuda()   # 多GPU

    model = model.cuda()

    criterion = nn.BCELoss(reduction='mean')  # 多分类用BCELoss
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)

    max_avg = 0

    writer = SummaryWriter(os.path.join('runs', 'fundus_' + model_name[:-4]))
    for epoch in range(START_EPOCH, EPOCHS):
        train(model, train_loader, optimizer, scheduler, criterion, writer, epoch)
        avg = validate(model, val_loader, criterion, writer, epoch)

        if avg > max_avg:
            torch.save(model, './model/fundus/' + model_name)
            max_avg = avg
    writer.close()


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    end = time.time()
    print('总耗时', end - start)
