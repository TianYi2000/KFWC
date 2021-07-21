import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class LabelRelevance(nn.Module):
    def __init__(self, class_num=11):
        super(LabelRelevance, self).__init__()
        self.class_num = class_num

    def forward(self, output, target):
        loss = 0
        target_list = target.cpu().numpy().tolist()
        for idx in range(len(target_list)):
            yi = target_list[idx]
            yi_one = list()
            yi_zero = list()
            for index in range(len(yi)):
                label = yi[index]
                if label == 0:
                    yi_zero.append(index)
                elif label == 1:
                    yi_one.append(index)
                else:
                    print('label error!!! Neither 0 nor 1')
            loss_yi = 0
            # num = Variable(torch.Tensor(len(yi_one)*len(yi_zero))).cuda()
            num = len(yi_one) * len(yi_zero)
            if num == 0:
                num = self.class_num
            for p in yi_one:
                for q in yi_zero:
                    loss_yi += torch.exp(output[idx][q] - output[idx][p])
            loss_yi = loss_yi/num
            loss += loss_yi

        return loss
