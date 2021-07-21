import csv
import os
import random


list_dir = '/home/hejiawen/datasets/xingtai_AMD/label/AMD_second/fundus/'
dest_dir = '/home/hejiawen/datasets/xingtai_AMD/label/AMD_second/fundus/'

train_label_path = os.path.join(list_dir, 'fundus_label.csv')
f = open(train_label_path, 'r')
reader = csv.reader(f)
data = list(reader)
title = data[0]
image_list = data[1:]
f.close()

# f = open(os.path.join(list_dir, 'val_label.csv'), 'r')
# reader = csv.reader(f)
# data = list(reader)
# image_list.extend(data[1:])
# f.close()
#
# f = open(os.path.join(list_dir, 'test_label.csv'), 'r')
# reader = csv.reader(f)
# data = list(reader)
# image_list.extend(data[1:])
# f.close()

print('length:', len(image_list))

random.shuffle(image_list)

with open(os.path.join(dest_dir, 'train_label.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    writer.writerows(image_list[:int(0.8 * len(image_list))])

with open(os.path.join(dest_dir, 'val_label.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    writer.writerows(image_list[int(0.8 * len(image_list)):int(0.9 * len(image_list))])

with open(os.path.join(dest_dir, 'test_label.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    writer.writerows(image_list[int(0.9 * len(image_list)):len(image_list)])

# f = open(os.path.join(dest_dir, 'OCT_label.csv'), 'r')
# list_1024 = list()
# for line in f:
#     line = line.strip()
#     line = line.split('.')[0]
#     list_1024.append(line)
# print('length 2:', len(list_1024))
# f.close()
#
# new_list = list()
# for line in image_list:
#     if line[0] in list_1024:
#         new_list.append(line)
# print('length 3:', len(new_list))
