import os
import time
import datetime
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from torchvision import transforms
from tqdm import tqdm

from data.base_dataset import Preproc, Rescale, ToTensor, Resize
from data.csv_dataset import Lesion_Complaint_Dataset
from utils.utils import calc_kappa
from utils.draw import draw_roc

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
    parser.add_argument('--two_stream_path', type=str, help='the model file path of two stream model', required=True)
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


def test(model, test_loader, criterion):
    model.eval()
    y_pred = []
    y_true = []
    tbar = tqdm(test_loader, desc='\r', ncols=100)  # 进度条
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
    print('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.
              format('f1', 'auroc', 'recall', 'precision', 'acc', 'kappa', 'loss'))
    print('{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.
              format(str(round(f1,4)), str(round(auroc,4)), str(round(recall,4)), str(round(precision,4)),
                     str(round(acc,4)), str(round(kappa,4)), str(round(out_loss,4)) ))

    return avg



def main():


    test_tf = transforms.Compose([
        Resize(args.oct_size),
        ToTensor(),
        transforms.Normalize(mean=mean[args.oct_size], std=std[args.oct_size])
    ])

    test_loader = torch.utils.data.DataLoader(
        Lesion_Complaint_Dataset(data_dir, 'test', test_tf, classCount, lesion_num, lesion_text, list_dir=list_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    model = torch.load(args.two_stream_path)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    avg = test(model, test_loader, criterion)

if __name__ == '__main__':
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
    data_dir = os.path.join(args.root_path, data_dir)
    list_dir = os.path.join(args.root_path, list_dir)
    # writer = SummaryWriter(os.path.join('runs', 'OCT/' + model_name[:-4]))
    print("Test Single Lesion Net ")
    start = time.time()
    main()
    end = time.time()
    print('Finish Single Lesion Net, Time=', end - start)

