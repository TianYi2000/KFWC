import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from resnest.torch import resnest50
from net.SCNet.scnet import scnet50

pre_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnest50": resnest50,
    "scnet50": scnet50,
    "inceptionv3": models.inception_v3,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19
}

def pretrain_models(model_name = 'resnet50', inner_feature=1000 ,lock_weight = False):
    model = pre_models[model_name](pretrained=True)

    if (lock_weight == True):
        for p in model.parameters():
            p.requires_grad = False
    if 'vgg' in model_name:
        model.classifier[3] = nn.Linear(in_features=4096, out_features=inner_feature, bias=True)
        model.classifier = model.classifier[:5]
    else:
        kernel_count = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
    return model

def load_models(model_path, model_name = 'resnet50',label_type ='single-label', inner_feature=1000 ,lock_weight = False):
    model = torch.load(model_path)
    if 'vgg' in model_name:
        model.classifier = model.classifier[:5]
    else:
        kernel_count = model.fc[0].in_features
    # if label_type == 'multilabel':
    #     model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
    # else:
        model.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
    return model

class TwoStreamNet(nn.Module):
    def __init__(self, fundus_path,OCT_path, fundus_model='resnest50', OCT_model='inceptionv3',
                 num_classes=1000, label_type='single-label', inner_feature=1000):

        super(TwoStreamNet, self).__init__()
        self.label_type = label_type
        self.model1 = load_models(model_path= fundus_path, model_name=fundus_model, label_type=label_type, inner_feature=inner_feature)
        self.model2 = load_models(model_path= OCT_path, model_name=OCT_model, label_type=label_type, inner_feature=inner_feature)
        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes), nn.Softmax(dim=1))

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



