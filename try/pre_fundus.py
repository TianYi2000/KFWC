import os
import cv2
import csv
import sys
sys.path.append('../')
from data.preprocessing import classification_preprocessing


data_dir = '/home/hejiawen/datasets/Messidor/image/'
list_dir = '/home/hejiawen/datasets/Messidor/label/'
dest_dir = '/home/hejiawen/datasets/Messidor_preprocessing'
phases = ['test', 'val', 'train']
pre_option = {
    'crop_flag': True,
    'required_size': 768,
    'factor': 0.9,
    'pre_flag': 1,
    'strength_factor': 30,
}

for phase in phases:
    label_path = os.path.join(list_dir, phase + '_label.csv')
    assert os.path.exists(label_path)

    image_list = list()
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]
        for line in data:
            image_list.append(line[0])

    for line in image_list:
        img = cv2.imread(os.path.join(data_dir, phase, line))
        out_dict = classification_preprocessing(img, pre_option, debug=False)
        if 'pre_img' in out_dict:
            pre_img = out_dict['pre_img']
            save_path = os.path.join(dest_dir, phase, line)
            try:
                cv2.imwrite(save_path, pre_img)
            except Exception as e:
                print('save error:', e)
    print(phase, 'OK!')




