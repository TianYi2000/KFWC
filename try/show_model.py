import sys
import torch
from torchvision import models
import torch.nn as nn
sys.path.append("..")
# from resnet1 import resnet34
# from net.SCNet.scnet import scnet50


# model = vgg11_bn()
# model_dict = model.state_dict()
# pretrained_dict = torch.load('../model/best_model_zhangkang.pth')
#
# # print(pretrained_dict)
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
#
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 16)
# for child in model.children():
#     print(child)
# model_path = '../model/OCT/best_model_zhangkang_1006_resnet34_pre_nolock.pth'
# classCount = 12
#
# model = models.resnet34(pretrained=True)
# kernelCount = model.fc.in_features
# model.fc = torch.nn.Sequential(torch.nn.Linear(kernelCount, classCount), torch.nn.Sigmoid())  # 多分类用sigmoid
# model_dict = model.state_dict()
# pretrained_dict = torch.load(model_path)
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# print(model.state_dict())
# for child in model.children():
#     print(child)
#     for param in child.parameters():
#         print(param.requires_grad)
        # param.requires_grad = False

from resnest.torch import resnest50
from net.SCNet.scnet import scnet50

model = models.resnet18(pretrained=True)
kernel_count = model.fc.in_features
print('resnet18:', kernel_count)

model = models.resnet34(pretrained=True)
kernel_count = model.fc.in_features
print('resnet34:', kernel_count)

model = models.resnet50(pretrained=True)
kernel_count = model.fc.in_features
print('resnet50:', kernel_count)

model = resnest50(pretrained=True)
kernel_count = model.fc.in_features
print('resnest50:', kernel_count)

model = models.inception_v3(pretrained=True)
kernel_count = model.AuxLogits.fc.in_features
print('inceptionv3:', kernel_count)
model.AuxLogits.fc = nn.Sequential(nn.Linear(kernel_count, 13), nn.Sigmoid())
kernel_count = model.fc.in_features
print('inceptionv3:', kernel_count)
model.fc = nn.Sequential(nn.Linear(kernel_count, 13), nn.Sigmoid())
model.aux_logits = False

model = scnet50(pretrained=True)
kernel_count = model.fc.in_features
print('scnet:', kernel_count)
