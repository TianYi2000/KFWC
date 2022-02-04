from torch.utils.data import Dataset
from os.path import join, exists
import numpy as np
from torch import from_numpy, sort
from PIL import Image
import cv2
import csv
from transformers import BertTokenizer


class ImageDataset(Dataset):
    def __init__(self, data_dir, phase, transforms, cla_num, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.cla_num = cla_num
        self.image_list = None
        self.label_list = None
        self.read_lists()   # 读取数据集

    def __getitem__(self, index):
        data = Image.open(join(self.data_dir, self.image_list[index]))
        data = data.convert('RGB')
        # data = data.crop((data.size[1], 0, data.size[0], data.size[1]))
        # if self.phase == 'train':
        #     data = imgtrans(data)
        data = self.transforms(data)
        label = from_numpy(np.array(self.label_list[index]))
        return tuple([data, label])

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        label_path = join(self.list_dir, self.phase + '_label.csv')
        # print(label_path)
        assert exists(label_path)

        with open(label_path, 'r',encoding = "utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            self.image_list = list()
            self.label_list = list()
            for line in data:
                label = list()
                for item in line[2: self.cla_num + 2]:
                    item = int(item)
                    label.append(int(item))
                self.image_list.append(line[0])
                self.label_list.append(label)
        assert len(self.image_list) == len(self.label_list)

        if self.phase == 'train':
            print('Total train image is : %d ' % len(self.image_list))
        elif self.phase == 'val':
            print('Total val pid is : %d ' % len(self.image_list))
        else:
            print('Total test pid is : %d ' % len(self.image_list))


class TwoStreamDataset(Dataset):
    def __init__(self, data_dir, phase, fundus_transforms, OCT_trainsforms, cla_num, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.fundus_transforms = fundus_transforms
        self.OCT_transforms = OCT_trainsforms
        self.cla_num = cla_num
        self.fundus_list = None
        self.OCT_list = None
        self.label_list = None
        self.read_lists()   # 读取数据集

    def __getitem__(self, index):
        fundus = Image.open(join(self.data_dir, self.fundus_list[index]))
        fundus = fundus.convert('RGB')
        fundus = self.fundus_transforms(fundus)

        OCT = Image.open(join(self.data_dir, self.OCT_list[index]))
        OCT = OCT.convert('RGB')
        OCT = self.OCT_transforms(OCT)
        label = from_numpy(np.array(self.label_list[index]))
        return tuple([fundus, OCT, label])

    def __len__(self):
        return len(self.fundus_list)

    def read_lists(self):
        label_path = join(self.list_dir, self.phase + '_label.csv')
        # print(label_path)
        assert exists(label_path)

        with open(label_path, 'r',encoding = "utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            self.fundus_list = list()
            self.OCT_list = list()
            self.label_list = list()
            for line in data:
                self.fundus_list.append(line[0])
                self.OCT_list.append(line[1])

                if line[2] == 'CNV':
                    self.label_list.append(0)   #[1,0,0]
                elif line[2] == 'PCV':
                    self.label_list.append(1)   #[0,1,0]
                elif line[2] == 'Non-wet-AMD':
                    self.label_list.append(2)   #[0,0,1]
                else:
                    print('label error:', line[2])
        assert len(self.fundus_list) == len(self.OCT_list) == len(self.label_list)

        if self.phase == 'train':
            print('Total train image is : %d ' % len(self.fundus_list))
        elif self.phase == 'val':
            print('Total val pid is : %d ' % len(self.fundus_list))
        else:
            print('Total test pid is : %d ' % len(self.fundus_list))


class MultiModeDataset(Dataset):
    def __init__(self, data_dir, phase, transforms, cla_num, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.cla_num = cla_num
        self.image_list = None
        self.sent_list = None
        self.label_list = None
        self.tokenizer = BertTokenizer.from_pretrained("/home/hejiawen/pytorch/wet_AMD_signs_multilabel_classification/"
                                                       "net/bert/bert-base-chinese/")
        self.read_lists()   # 读取数据集

    def __getitem__(self, index):
        images = Image.open(join(self.data_dir, self.image_list[index]))
        images = images.convert('RGB')
        images = self.transforms(images)
        captions = from_numpy(np.array(self.sent_list(index)))
        cap_len = from_numpy(np.array(len(captions)))
        label = from_numpy(np.array(self.label_list[index]))
        
        return tuple([images, captions, cap_len, label])

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        label_path = join(self.list_dir, self.phase + '_label.csv')
        # print(label_path)
        assert exists(label_path)

        with open(label_path, 'r',encoding = "utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            self.image_list = list()
            self.sent_list = list()
            self.label_list = list()
            for line in data:
                label = list()
                for item in line[2: self.cla_num + 2]:
                    item = int(item)
                    label.append(item)
                self.image_list.append(line[0])
                self.label_list.append(label)
                tokens = self.tokenizer.tokenize(line[self.cla_num + 2])
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                self.sent_list.append(tokens)
        assert len(self.image_list) == len(self.sent_list) == len(self.label_list)

        if self.phase == 'train':
            print('Total train image is : %d ' % len(self.image_list))
        elif self.phase == 'val':
            print('Total val pid is : %d ' % len(self.image_list))
        else:
            print('Total test pid is : %d ' % len(self.image_list))


def prepare_data(data):
    images, captions, cap_len, target = data
    sorted_cap_lens, sorted_cap_indices = sort(cap_len, 0, True)

    images = images[sorted_cap_indices]
    captions = captions[sorted_cap_indices].squeeze()
    target = target[sorted_cap_indices]

    return images, captions, sorted_cap_lens, target

import torch
if __name__ == '__main__':
    test1 = torch.empty(1)
    test2 = torch.empty(2)
    test3 = torch.empty([2,2])
    test4 = torch.empty([3,4])
    test5 = torch.empty(3,4)
    print(test1)
    print(test2)
    print(test3)
    print(test4)
    print(test5)