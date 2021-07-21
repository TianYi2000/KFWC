import os
import csv
import random


def match_two_stream_data(OCT_dir, fundus_dir, dest_dir):
    eyes = dict()
    with open(OCT_dir, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]
        for line in data:
            if line[12]:
                patient = line[12]
            else:
                patient = line[13].split('_')[0]
            key = patient + line[15] + line[1]
            if key not in eyes:
                eyes[key] = list()
            eyes[key].append(line[0])

    two_stream_data = dict()
    with open(fundus_dir, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]
        for line in data:
            if line[7]:
                patient = line[7]
            else:
                patient = line[8].split('_')[0]
            key = patient + line[10] + line[1]
            if key in eyes:
                for url in eyes[key]:
                    new_data = [line[0], url, line[1], patient, line[10]]
                    if key not in two_stream_data:
                        two_stream_data[key] = list()
                    two_stream_data[key].append(new_data)

    print('num of eyes:', len(two_stream_data))

    keys = list(two_stream_data.keys())
    random.shuffle(keys)

    new_data = {'train': list(), 'val': list(), 'test': list()}

    num = 0
    for key in keys:
        if num < 15:
            for data in two_stream_data[key]:
                new_data['val'].append(data)
        elif num < 30:
            for data in two_stream_data[key]:
                new_data['test'].append(data)
        else:
            for data in two_stream_data[key]:
                new_data['train'].append(data)
        num += 1

    new_title = ['fundus_url', 'OCT_url', 'label', 'patient', 'eye_pose']

    for phase in ['train', 'val', 'test']:
        print('length of', phase, ':', len(new_data[phase]))
        random.shuffle(new_data[phase])
        with open(os.path.join(dest_dir, phase + '_label.csv'), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_title)
            for data in new_data[phase]:
                writer.writerow(data)


def match_multi_label(OCT_dir, fundus_dir, two_stream_dir):
    with open(OCT_dir, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        OCT_data = list(reader)
        OCT_title = OCT_data[0]
        OCT_data = OCT_data[1:]

    with open(fundus_dir, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        fundus_data = list(reader)
        fundus_title = fundus_data[0]
        fundus_data = fundus_data[1:]

    new_OCT_data_dict = dict()
    new_fundus_data_dict = dict()
    for phase in ['train', 'val', 'test']:
        with open(os.path.join(two_stream_dir, phase + '_label.csv'), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]

        new_OCT_data = list()
        for line in data:
            url = line[1]
            for OCT_line in OCT_data:
                if OCT_line[0] == url and OCT_line not in new_OCT_data:
                    new_OCT_data.append(OCT_line)
                    OCT_data.remove(OCT_line)
                    break
        new_OCT_data_dict[phase] = new_OCT_data.copy()
        print('length of OCT', phase, ':', len(new_OCT_data))

        new_fundus_data = list()
        for line in data:
            url = line[0]
            for fundus_line in fundus_data:
                if fundus_line[0] == url and fundus_line not in new_fundus_data:
                    new_fundus_data.append(fundus_line)
                    break
        new_fundus_data_dict[phase] = new_fundus_data.copy()
        print('length of fundus', phase, ':', len(new_fundus_data))

    print('length of OCT_data:', len(OCT_data))
    print('length of fundus_data:', len(fundus_data))

    random.shuffle(OCT_data)
    random.shuffle(fundus_data)

    new_OCT_data_dict['train'].extend(OCT_data[:1837])
    new_OCT_data_dict['val'].extend(OCT_data[1837:2067])
    new_OCT_data_dict['test'].extend(OCT_data[2067:])

    new_fundus_data_dict['train'].extend(fundus_data[:662])
    new_fundus_data_dict['val'].extend(fundus_data[662:745])
    new_fundus_data_dict['test'].extend(fundus_data[745:])

    for phase in ['train', 'val', 'test']:
        print('length of OCT', phase, ':', len(new_OCT_data_dict[phase]))
        print('length of fundus', phase, ':', len(new_fundus_data_dict[phase]))
        with open(os.path.join(two_stream_dir, 'OCT', phase + '_label.csv'), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(OCT_title)
            for line in new_OCT_data_dict[phase]:
                writer.writerow(line)

        with open(os.path.join(two_stream_dir, 'fundus', phase + '_label.csv'), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(fundus_title)
            for line in new_fundus_data_dict[phase]:
                writer.writerow(line)


if __name__ == '__main__':
    OCT_dir = '/home/hejiawen/datasets/xingtai_AMD/label/AMD_second/OCT/OCT_label.csv'
    fundus_dir = '/home/hejiawen/datasets/xingtai_AMD/label/AMD_second/fundus/fundus_label.csv'

    # dest_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/'
    # match_two_stream_data(OCT_dir, fundus_dir, dest_dir)

    two_stream_dir = '/home/hejiawen/datasets/AMD_processed/label/new_two_stream/'
    match_multi_label(OCT_dir, fundus_dir, two_stream_dir)
