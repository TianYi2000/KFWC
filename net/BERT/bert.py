import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')


bert_path = "./net/bert/bert-base-chinese/"    # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path)   # 初始化分词器

class Bert_Model(nn.Module):
    def __init__(self):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)     # 加载预训练模型权重
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类


        # for name ,param in self.bert.named_parameters():
        #     param.requires_grad = False


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]   # 池化后的输出 [bs, config.hidden_size]
        # logit = self.fc(out_pool)   #  [bs, classes]
        x = out_pool.detach()  #锁Bert权重
        return x