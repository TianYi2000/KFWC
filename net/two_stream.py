import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from net.SCNet.scnet import scnet50

def pretrain_models(model_name = 'resnet50', inner_feature=1000 ,lock_weight = False):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)

    # elif model_name == "resnest50":
    #     from resnest.torch import resnest50
    #     model = resnest50(pretrained=True)

    elif model_name == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        kernel_count = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        model.aux_logits = False

    elif model_name == 'scnet50':
        from net.SCNet.scnet import scnet50
        model = scnet50(pretrained=True)
    else:
        return

    if (lock_weight == True):
        for p in model.parameters():
            p.requires_grad = False
    kernel_count = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
    return model

def load_models(model_path, model_name = 'resnet50',label_type ='single-label', inner_feature=1000 ,lock_weight = False):
    model = torch.load(model_path)
    if (lock_weight):
        for p in model.parameters():
            p.requires_grad = False
    if 'resnet' in model_name and '50' not in model_name:
        kernel_count = 512     # 读出来的---------------

    elif 'resnest' in model_name or 'scnet' in model_name or 'resnet50' in model_name:
        kernel_count = 2048  # 读出来的---------------

    elif 'inception' in model_name:
        kernel_count = 768     # 读出来的---------------
        if label_type == 'multilabel':
            model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
        else:
            model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        kernel_count = 2048     # 读出来的---------------
        model.aux_logits = False

    # todo(hty):这里的nn.Linear(kernel_count, inner_feature)是否有办法赋予初始参数（而非全0或者是自带的某些默认初始参数）
    if label_type == 'multilabel':
        model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
    else:
        model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))

    return model

class TwoStreamNet(nn.Module):
    def __init__(self, fundus_path,OCT_path, fundus_model='resnest50', OCT_model='inceptionv3',
                 num_classes=1000, label_type='single-label', inner_feature=1000):

        super(TwoStreamNet, self).__init__()
        self.label_type = label_type

        self.model1 = load_models(model_path= fundus_path, model_name=fundus_model, label_type=label_type, inner_feature=inner_feature)
        self.model2 = load_models(model_path= OCT_path, model_name=OCT_model, label_type=label_type, inner_feature=inner_feature)
        #todo(hty):这里只有一层会不会不太够？
        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes))

    def forward(self, x1, x2):

        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = self.fc(torch.cat((x1, x2), 1))
        return x

class Only_Fundus_Net(nn.Module):
    def __init__(self, fundus_path, fundus_model='resnest50', OCT_model='inceptionv3',
                 num_classes=1000, label_type='single-label', inner_feature=1000):

        super(Only_Fundus_Net, self).__init__()
        self.label_type = label_type

        self.model1 = load_models(model_path= fundus_path, model_name=fundus_model, label_type=label_type, inner_feature=inner_feature)
        self.model2 = pretrain_models(model_name = OCT_model, inner_feature = inner_feature, lock_weight = False)
        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes))

    def forward(self, x1, x2):

        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = self.fc(torch.cat((x1, x2), 1))
        return x

class Only_OCT_Net(nn.Module):
    def __init__(self, OCT_path, fundus_model='resnest50', OCT_model='inceptionv3',
                 num_classes=1000, label_type='single-label', inner_feature=1000):

        super(Only_OCT_Net, self).__init__()
        self.label_type = label_type

        self.model1 = pretrain_models(model_name = fundus_model, inner_feature = inner_feature, lock_weight = False)
        self.model2 = load_models(model_path= OCT_path, model_name=OCT_model, label_type=label_type, inner_feature=inner_feature)
        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes))

    def forward(self, x1, x2):

        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = self.fc(torch.cat((x1, x2), 1))
        return x

class BaseLineNet(nn.Module):
    def __init__(self, fundus_model='resnest50', OCT_model='inceptionv3', num_classes=1000,
                 label_type='single-label', inner_feature=1000):

        super(BaseLineNet, self).__init__()
        self.label_type = label_type

        self.model1 = pretrain_models(model_name = fundus_model, inner_feature = inner_feature, lock_weight = False)
        self.model2 = pretrain_models(model_name = OCT_model, inner_feature = inner_feature, lock_weight = False)
        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes))

    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = self.fc(torch.cat((x1, x2), 1))
        return x



