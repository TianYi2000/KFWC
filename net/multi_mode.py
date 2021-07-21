import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models


class RnnEncoder(nn.Module):
    def __init__(self, words_num, feature_dim=256, drop_prob=0.5, hidden_dim=128,
                 layers_num=1, bidirectional=True, rnn_type='LSTM'):
        super(RnnEncoder, self).__init__()
        # self.n_steps = 18   # 没用
        self.words_num = words_num  # 一共有多少词
        self.feature_dim = feature_dim  # 输出向量的维度，可以256，或者参考图像的维度
        self.drop_prob = drop_prob  # 0.5
        self.layers_num = layers_num  # 一共多少层，用1
        self.bidirectional = bidirectional  # 是否双向，用双向
        self.rnn_type = rnn_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.hidden_dim = hidden_dim // self.num_directions   # 隐藏层特征维度

        self.encoder = None
        self.drop = None
        self.rnn = None
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.words_num, self.feature_dim)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.feature_dim, self.hidden_dim, self.layers_num, batch_first=True,
                               dropout=self.drop_prob, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.feature_dim, self.hidden_dim, self.layers_num, batch_first=True,
                              dropout=self.drop_prob, bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.layers_num * self.num_directions, batch_size, self.hidden_dim).zero_()),
                    Variable(weight.new(self.layers_num * self.num_directions, batch_size, self.hidden_dim).zero_()))
        elif self.rnn_type == 'GRU':
            return Variable(weight.new(self.layers_num * self.num_directions, batch_size, self.hidden_dim).zero_())
        else:
            raise NotImplementedError

    # captions是向量，list，需要从长到短排序，后面补0。prepare_data里
    # cap_lens是每一个caption的长度
    # hidden是初始的hidden
    def forward(self, captions, cap_lens, hidden):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = nn.utils.rnn.pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        # output = pad_packed_sequence(output, batch_first=True)[0]
        # words_emb = output.transpose(1, 2)
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.hidden_dim * self.num_directions)
        return sent_emb  # 只需要sent_emb


class CnnEncoder(nn.Module):
    def __init__(self, model_type='resnest50', feature_dim=1000, label_type='multi_label'):

        super(CnnEncoder, self).__init__()
        self.label_type = label_type

        if model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_type == 'resnet34':
            self.model = models.resnet34(pretrained=True)
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif MODEL == "resnest50":
            from resnest.torch import resnest50
            self.model = resnest50(pretrained=True)
        elif MODEL == "inceptionv3":
            self.model = models.inception_v3(pretrained=True, aux_logits=False)
        elif MODEL == 'scnet50':
            from net.SCNet.scnet import scnet50
            self.model = scnet50(pretrained=True)
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=True)

        if 'vgg' in model_type:
            if self.label_type == 'multi_label':
                self.fc = nn.Sequential(nn.Linear(1000, feature_dim), nn.Sigmoid())  # kernelCount
            else:
                self.fc = nn.Sequential(nn.Linear(1000, feature_dim))
        else:
            kernel_count = self.model.fc.in_features
            self.model.fc = nn.Linear(kernel_count, kernel_count)
            if self.label_type == 'multi_label':
                self.fc = nn.Sequential(nn.Linear(kernel_count, feature_dim), nn.Sigmoid())  # kernelCount
            else:
                self.fc = nn.Sequential(nn.Linear(kernel_count, feature_dim))  # kernelCount 改成500 就对了。

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class MultiMode(nn.Module):
    def __init__(self, cnn_model, rnn_model, class_num, feature_dim=1256, label_type='multi_label'):
        super(MultiMode, self).__init__()
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.feature_dim = feature_dim
        if label_type == 'multi_label':
            self.fc = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(feature_dim, class_num))

    def forward(self, image, captions, cap_lens, batch_size):
        image_feature = self.cnn_model(image)
        hidden = self.rnn_model.init_hidden(batch_size)
        sent_feature = self.rnn_model(captions, cap_lens, hidden)
        x = self.fc(torch.cat((image_feature, sent_feature), 1))
        return x
