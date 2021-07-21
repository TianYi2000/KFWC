import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize
from data.csv_dataset import MultiModeDataset, prepare_data
from net.multi_mode import RnnEncoder, CnnEncoder, MultiMode
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

METHOD = ''
CNN_MODEL = "scnet50"
RNN_MODEL = 'LSTM'
LOSS = 'bceloss'

START_EPOCH = 0
EPOCHS = 500

MOMENTUM = 0.9  # RECEIVED_PARAMS["momentum"]
WEIGHT_DECAY = 0.0001  # RECEIVED_PARAMS["weight_decay"]
LR = 0.001  # RECEIVED_PARAMS["learning_rate"]
NAME = METHOD + "+" + str(EPOCHS) + "+" + str(LR) + '+' + LOSS

RESUME = False
model_path = ''
model_name = '2021_04_16+' + CNN_MODEL + '+' + RNN_MODEL + '+' + NAME + '.pth'
# model_name = 'try_loss.pth'
print("Train Multi-Mode-OCT", model_name, 'RESUME:', RESUME)

BATCH_SIZE = 8  # RECEIVED_PARAMS["batch_size"]
WORKERS = 1

IMAGE_SIZE = 224  # 224 for resnet, 299 for inception
WORDS_NUM = 21128   # tokenizer.vocab_size

AVERAGE = 'weighted'

cols = ['视网膜内液性暗腔', '视网膜下积液', 'RPE脱离', 'RPE下高反射病灶', '视网膜内或视网膜下高反射病灶', '尖锐的RPED峰',
        '双层征', '多发性RPED', 'RPED切迹', '视网膜内高反射硬性渗出']
classCount = len(cols)
samples_per_cls = [445, 543, 808, 109, 1090, 23, 241, 88, 21, 158]

# 训练的df 路径
data_dir = '/home/hejiawen/datasets/AMD_processed/'
list_dir = '/home/hejiawen/datasets/xingtai_AMD/label/AMD_second/OCT/'


def train(multi_modal, train_loader, optimizer, scheduler, criterion, writer, epoch):
    multi_modal.train()

    tbar = tqdm(train_loader, desc='\r', ncols=100)  # 进度条
    y_pred = []
    y_true = []
    loss_val = 0
    loss_valnorm = 0
    print(tbar)
    for batch_idx, data in enumerate(tbar):
        images, captions, cap_len, target = prepare_data(data)
        target = target.float()
        images = images.cuda()
        captions = captions.cuda()
        cap_len = cap_len.cuda()
        target = target.cuda()
        
        optimizer.zero_grad()
        output = multi_modal(images, captions, cap_len, BATCH_SIZE)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        writer.add_scalar("Train/loss", loss.item(), epoch * len(train_loader) + batch_idx)
        tbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

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


def validate(multi_modal, val_loader, criterion, writer, epoch):
    multi_modal.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(val_loader, desc='\r', ncols=100)
    loss_val = 0
    loss_valnorm = 0
    for batch_idx, data in enumerate(tbar):
        images, captions, cap_len, target = prepare_data(data)
        target = target.float()
        images = images.cuda()
        captions = captions.cuda()
        cap_len = cap_len.cuda()
        target = target.cuda()

        output = multi_modal(images, captions, cap_len, BATCH_SIZE)

        loss = criterion(output, target)

        y_pred.extend(output.data.cpu().numpy())
        y_true.extend(target.data.cpu().numpy())
        writer.add_scalar("Val/loss", loss.item(), epoch * len(val_loader) + batch_idx)
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
    # 通过随机变化来进行数据增强
    train_tf = transforms.Compose([
        Preproc(0.2),
        # Rescale(IMAGE_SIZE),    # 等比例缩小
        # transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪，fundus适用，OCT不适用
        Resize(IMAGE_SIZE),  # 非等比例缩小
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        Preproc(0.2),
        # Rescale(IMAGE_SIZE),    # 等比例缩小
        # transforms.CenterCrop(IMAGE_SIZE),  # 以中心裁剪
        Resize(IMAGE_SIZE),     # 非等比例缩小
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        MultiModeDataset(data_dir, 'train', train_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        MultiModeDataset(data_dir, 'val', val_tf, classCount, list_dir=list_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )

    image_encoder = CnnEncoder(model_type=CNN_MODEL, num_classes=classCount)
    text_encoder = RnnEncoder(WORDS_NUM, rnn_type=RNN_MODEL)  # 预处理所有文本后得到words_num

    if RESUME:
        model_dict = image_encoder.state_dict()
        pretrained_dict = torch.load(model_path)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
        model_dict.update(pretrained_dict)
        image_encoder.load_state_dict(model_dict)

    image_encoder.input_space = 'RGB'
    image_encoder.input_size = [3, IMAGE_SIZE, IMAGE_SIZE]
    image_encoder.input_range = [0, 1]
    image_encoder.mean = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5] for inception* networks
    image_encoder.std = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5] for inception* networks

    multi_modal = MultiMode(image_encoder, text_encoder, classCount)
    # model = nn.DataParallel(model).cuda()   # 多GPU
    multi_modal = multi_modal.cuda()

    criterion = nn.BCELoss(reduction='mean')  # 多分类用BCELoss
    optimizer = optim.SGD(multi_modal.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)

    max_avg = 0

    writer = SummaryWriter(os.path.join('runs', 'multi-mode-OCT_' + model_name[:-4]))
    for epoch in range(START_EPOCH, EPOCHS):
        train(multi_modal, train_loader, optimizer, scheduler, criterion, writer, epoch)
        avg = validate(multi_modal, val_loader, criterion, writer, epoch)

        if avg > max_avg:
            torch.save(model, './model/multi-mode-OCT/' + model_name)
            max_avg = avg
    writer.close()


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    end = time.time()
    print('总耗时', end - start)
