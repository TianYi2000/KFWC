from torch.utils.data import Dataset
from os.path import join, exists
import numpy as np
from torch import from_numpy, sort
from PIL import Image
import cv2
import csv
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
tokenizer = BertTokenizer.from_pretrained(("./net/BERT/bert-base-chinese/"))


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

class Lesion_Complaint_Dataset(Dataset):
    def __init__(self, data_dir, phase, transforms, cla_num, lesion_num, lesion_text ,list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.cla_num = cla_num
        self.lesion_num = lesion_num
        self.lesion_text= lesion_text
        self.image_list = None
        self.lesion_list = None
        self.lesion_tokens = None
        self.complaint_tokens = None

        self.label_list = None
        self.read_lists()   # 读取数据集

    def __getitem__(self, index):
        data = Image.open(join(self.data_dir, self.image_list[index]))
        data = data.convert('RGB')
        data = self.transforms(data)
        lesion_token = self.lesion_tokens[index]
        complaint_token = self.complaint_tokens[index]

        lesion_id = torch.LongTensor(lesion_token[0])
        lesion_mask = torch.LongTensor(lesion_token[1])
        lesion_type = torch.LongTensor(lesion_token[2])

        complaint_id = torch.LongTensor(complaint_token[0])
        complaint_mask = torch.LongTensor(complaint_token[1])
        complaint_type = torch.LongTensor(complaint_token[2])

        label = from_numpy(np.array(self.label_list[index]))
        return tuple([data, lesion_id, lesion_mask, lesion_type, complaint_id, complaint_mask, complaint_type, label])

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        label_path = join(self.list_dir, self.phase + '_label.csv')
        # print(label_path)
        assert exists(label_path)
        maxlen = 50
        with open(label_path, 'r',encoding = "utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            self.image_list = list()
            self.lesion_list = list()
            self.lesion_tokens = list()
            self.complaint_tokens = list()
            self.label_list = list()
            for line in data:
                self.image_list.append(line[0])

                if line[1] == 'CNV':
                    self.label_list.append(0)   #[1,0,0]
                elif line[1] == 'PCV':
                    self.label_list.append(1)   #[0,1,0]
                elif line[1] == 'Non-wet-AMD':
                    self.label_list.append(2)   #[0,0,1]
                else:
                    print('label error:', line[1])

                lesion = list()
                for item in line[2: self.lesion_num + 2]:
                    item = int(item)
                    lesion.append(int(item))
                self.lesion_list.append(lesion)

                lesion_text = ''
                for index, exist in enumerate(lesion):
                    if exist == 1:
                        if lesion_text != '':
                            lesion_text = lesion_text + ';'
                        lesion_text = lesion_text + self.lesion_text[index]

                complaint_text = line[14]

                if len(lesion_text) > maxlen or len(complaint_text) > maxlen:
                    print('maxlen is not enough!')

                lesion_token = list()
                encode_dict = tokenizer.encode_plus(text=lesion_text, max_length=maxlen,
                                            padding='max_length', truncation=True)
                lesion_token.append(encode_dict['input_ids'])
                lesion_token.append(encode_dict['attention_mask'])
                lesion_token.append(encode_dict['token_type_ids'])
                self.lesion_tokens.append(lesion_token)

                complaint_token = list()
                encode_dict = tokenizer.encode_plus(text=complaint_text, max_length=maxlen,
                                            padding='max_length', truncation=True)
                complaint_token.append(encode_dict['input_ids'])
                complaint_token.append(encode_dict['attention_mask'])
                complaint_token.append(encode_dict['token_type_ids'])
                self.complaint_tokens.append(complaint_token)

        assert len(self.image_list) == len(self.label_list)

        if self.phase == 'train':
            print('Total train pid is : %d ' % len(self.image_list))
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

        id = torch.LongTensor(self.input_ids[index])
        mask = torch.LongTensor(self.input_masks[index])
        type = torch.LongTensor(self.input_types[index])

        label = from_numpy(np.array(self.label_list[index]))
        return tuple([fundus, OCT, id, mask, type, label])

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
            self.label_list = list()
            self.fundus_list = list()
            self.OCT_list = list()
            self.input_ids = list()
            self.input_types = list()
            self.input_masks = list()
            maxlen = 30
            for line in data:
                if line[0] == '新生血管性AMD':
                    self.label_list.append(0)   #[1,0,0]
                elif line[0] == 'PCV':
                    self.label_list.append(1)   #[0,1,0]
                elif line[0] == '非湿性AMD':
                    self.label_list.append(2)   #[0,0,1]
                else:
                    print('label error:', line[0])

                self.fundus_list.append(line[1])
                self.OCT_list.append(line[2])

                text = line[5]
                encode_dict = tokenizer.encode_plus(text=text, max_length=maxlen,
                                            padding='max_length', truncation=True)

                self.input_ids.append(encode_dict['input_ids'])
                self.input_types.append(encode_dict['token_type_ids'])
                self.input_masks.append(encode_dict['attention_mask'])

        assert len(self.fundus_list) == len(self.input_ids) == len(self.label_list)

        if self.phase == 'train':
            print('Total train pid is : %d ' % len(self.fundus_list))
        elif self.phase == 'val':
            print('Total val pid is : %d ' % len(self.fundus_list))
        else:
            print('Total test pid is : %d ' % len(self.fundus_list))


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