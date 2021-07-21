import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from resnest.torch import resnest50
from net.SCNet.scnet import scnet50


class TwoStreamNet(nn.Module):
    def __init__(self, fundus_path, OCT_path, fundus_model='resnest50', OCT_model='inceptionv3',
                 num_classes=1000, label_type='single-label', inner_feature=1000):

        super(TwoStreamNet, self).__init__()
        self.label_type = label_type

        self.model1 = torch.load(fundus_path)
        self.model2 = torch.load(OCT_path)
        # for p in self.model1.parameters():
        #     p.requires_grad = False
        #
        # for p in self.model2.parameters():
        #     p.requires_grad = False

        if 'resnet' in fundus_model and '50' not in fundus_model:
            kernel_count = 512     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())  # kernelCount
            else:
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif 'resnest' in fundus_model or 'scnet' in fundus_model or 'resnet50' in fundus_model:
            kernel_count = 2048  # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())  # kernelCount
            else:
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif 'inception' in fundus_model:
            kernel_count = 768     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model1.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
            else:
                self.model1.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            kernel_count = 2048     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
            else:
                self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            self.model1.aux_logits = False

        if 'resnet' in OCT_model and '50' not in OCT_model:
            kernel_count = 512     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())  # kernelCount
            else:
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif 'resnest' in OCT_model or 'scnet' in OCT_model or 'resnet50' in OCT_model:
            kernel_count = 2048  # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())  # kernelCount
            else:
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif 'inception' in OCT_model:
            kernel_count = 768     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model2.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
            else:
                self.model2.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            kernel_count = 2048     # 读出来的---------------
            if self.label_type == 'multilabel':
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature), nn.Sigmoid())
            else:
                self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            self.model2.aux_logits = False

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

        if fundus_model == "resnet18":
            self.model1 = models.resnet18(pretrained=True)
            # for p in self.model1.parameters():
            #     p.requires_grad = False
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif fundus_model == "resnet34":
            self.model1 = models.resnet34(pretrained=True)
            # for p in self.model1.parameters():
            #     p.requires_grad = False
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif fundus_model == "resnet50":
            self.model1 = models.resnet50(pretrained=True)
            # for p in self.model1.parameters():
            #     p.requires_grad = False
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif fundus_model == "resnest50":
            from resnest.torch import resnest50
            self.model1 = resnest50(pretrained=True)
            for p in self.model1.parameters():
                p.requires_grad = False
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif fundus_model == "inceptionv3":
            self.model1 = models.inception_v3(pretrained=True)
            # for p in self.model1.parameters():
            #     p.requires_grad = False
            kernel_count = self.model1.AuxLogits.fc.in_features
            self.model1.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            self.model1.aux_logits = False
        elif fundus_model == 'scnet50':
            from net.SCNet.scnet import scnet50
            self.model1 = scnet50(pretrained=True)
            for p in self.model1.parameters():
                p.requires_grad = False
            kernel_count = self.model1.fc.in_features
            self.model1.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        else:
            return

        if OCT_model == "resnet18":
            self.model2 = models.resnet18(pretrained=True)
            # for p in self.model2.parameters():
            #     p.requires_grad = False
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif OCT_model == "resnet34":
            self.model2 = models.resnet34(pretrained=True)
            # for p in self.model2.parameters():
            #     p.requires_grad = False
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif OCT_model == "resnet50":
            self.model2 = models.resnet50(pretrained=True)
            # for p in self.model2.parameters():
            #     p.requires_grad = False
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif OCT_model == "resnest50":
            from resnest.torch import resnest50
            self.model2 = resnest50(pretrained=True)
            for p in self.model2.parameters():
                p.requires_grad = False
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        elif OCT_model == "inceptionv3":
            self.model2 = models.inception_v3(pretrained=True)
            # for p in self.model2.parameters():
            #     p.requires_grad = False
            kernel_count = self.model2.AuxLogits.fc.in_features
            self.model2.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
            self.model2.aux_logits = False
        elif OCT_model == 'scnet50':
            from net.SCNet.scnet import scnet50
            self.model2 = scnet50(pretrained=True)
            for p in self.model2.parameters():
                p.requires_grad = False
            kernel_count = self.model2.fc.in_features
            self.model2.fc = nn.Sequential(nn.Linear(kernel_count, inner_feature))
        else:
            return

        self.fc = nn.Sequential(nn.Linear(inner_feature * 2, num_classes))

    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = self.fc(torch.cat((x1, x2), 1))
        return x



