import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

EDGE_RANGE = 0.2 #取20%的边界元素作为图的边界，用于判断黑色/白色边界

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


class Preproc(object):
    def __init__(self, sigma):
        #初始定义sigma参数
        self.sigma = sigma

    def __call__(self, sample):
        w, h = sample.size
        sample_numpy = np.array(sample)

        #判断图片周围EDGE_RANGE一圈是黑色还是白色，若为黑色需要将图片反色
        inverse = False
        top_edge = sample_numpy[0:int(h * EDGE_RANGE), : , 0].flatten()
        down_edge = sample_numpy[int(h * (1 - EDGE_RANGE)):(h - 1), : , 0].flatten()
        left_edge = sample_numpy[:, 0:int(w * EDGE_RANGE), 0].flatten()
        right_edge = sample_numpy[:, int(w * (1-EDGE_RANGE)):(w - 1), 0].flatten()
        edge_numpy = np.concatenate((top_edge, down_edge, left_edge, right_edge))
        if (np.mean(edge_numpy) < 255/2.0):
            sample = ImageOps.invert(sample)
            sample_numpy = np.array(sample)
            inverse = True

        mean = np.mean(sample_numpy)
        std = np.std(sample_numpy)
        threshold = mean + std * self.sigma
        # print(mean, std, threshold)
        # Top to Bottom
        top_index = 0
        for index in range(int(h/2)):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                top_index = index + 1
            else:
                break
        # Bottom to Top
        bottom_index = h-1
        for index in range(h-1, int(h/2), -1):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                bottom_index = index - 1
            else:
                break
        # Left to Right
        left_index = 0
        for index in range(int(w/2)):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                left_index = index + 1
            else:
                break
        # Right to Left
        right_index = w - 1
        for index in range(w - 1, int(w/2), -1):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                right_index = index - 1
            else:
                break
        # print(top_index,bottom_index+1, left_index,right_index+1)
        sample_numpy = sample_numpy[top_index:bottom_index+1, left_index:right_index+1]

        result_image = Image.fromarray(sample_numpy)
        if inverse:
            result_image = ImageOps.invert(result_image)
        return result_image


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):



        w, h = sample.size
        sample_numpy = np.array(sample)

        #判断图片周围EDGE_RANGE一圈是黑色还是白色，若为黑色需要将图片反色
        inverse = False
        top_edge = sample_numpy[0:int(h * EDGE_RANGE), : , 0].flatten()
        down_edge = sample_numpy[int(h * (1 - EDGE_RANGE)):(h - 1), : , 0].flatten()
        left_edge = sample_numpy[:, 0:int(w * EDGE_RANGE), 0].flatten()
        right_edge = sample_numpy[:, int(w * (1-EDGE_RANGE)):(w - 1), 0].flatten()
        edge_numpy = np.concatenate((top_edge, down_edge, left_edge, right_edge))
        if (np.mean(edge_numpy) < 255/2.0):
            sample_numpy = 1 - sample_numpy
            inverse = True

        mean = np.mean(sample_numpy)
        std = np.std(sample_numpy)
        threshold = mean + std*0.2
        print(mean, std, threshold)
        # Top to Bottom
        top_index = 0
        for index in range(int(h/2)):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                top_index = index + 1
            else:
                break
        # Bottom to Top
        bottom_index = h-1
        for index in range(h-1, int(h/2), -1):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                bottom_index = index - 1
            else:
                break
        # Left to Right
        left_index = 0
        for index in range(int(w/2)):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                left_index = index + 1
            else:
                break
        # Right to Left
        right_index = w - 1
        for index in range(w - 1, int(w/2), -1):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                right_index = index - 1
            else:
                break
        # print(top_index,bottom_index+1, left_index,right_index+1)
        sample_numpy = sample_numpy[top_index:bottom_index+1, left_index:right_index+1]

        result_image = Image.fromarray(sample_numpy)
        if inverse:
            result_image = Image.fromarray(1 - sample_numpy)
        return result_image


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = int(self.output_size), int(self.output_size)

        sample = sample.resize((new_h, new_w), Image.BICUBIC)

        return sample

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.size
        new_h, new_w = self.output_size

        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        sample = sample.crop((top, left, top + new_h, left + new_w))

        return sample


class ToTensor(object):

    def __call__(self, sample):
        input_image = np.array(sample, np.float32) / 255.0

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = input_image.transpose((2, 0, 1))
        return torch.from_numpy(input_image)

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample):
        sample = (sample - self.mean) / self.std

        return sample


class ImgTrans(object):
    def __call__(self, x):
        """
        input is a PIL image
        image data augmentation
        return also a PIL image
        """
        # random do the input augmentation
        if np.random.random() > 0.5:
            # sharpness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Sharpness(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # color augument
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Color(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # contrast augument
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Contrast(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # brightness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Brightness(x)
            x = enhancer.enhance(factor)
        return x

def cut (sample):
    w, h = sample.size
    sample_numpy = np.array(sample)

    #判断图片周围EDGE_RANGE一圈是黑色还是白色，若为黑色需要将图片反色
    inverse = False
    top_edge = sample_numpy[0:int(h * EDGE_RANGE), : , 0].flatten()
    down_edge = sample_numpy[int(h * (1 - EDGE_RANGE)):(h - 1), : , 0].flatten()
    left_edge = sample_numpy[:, 0:int(w * EDGE_RANGE), 0].flatten()
    right_edge = sample_numpy[:, int(w * (1-EDGE_RANGE)):(w - 1), 0].flatten()
    edge_numpy = np.concatenate((top_edge, down_edge, left_edge, right_edge))
    if (np.mean(edge_numpy) < 255/2.0):
        sample_numpy = 1 - sample_numpy
        inverse = True

    mean = np.mean(sample_numpy)
    std = np.std(sample_numpy)
    threshold = mean + std*0.2
    print(mean, std, threshold)
    # Top to Bottom
    top_index = 0
    for index in range(int(h/2)):
        if np.mean(sample_numpy[index, :, 0]) > threshold:
            top_index = index + 1
        else:
            break
    # Bottom to Top
    bottom_index = h-1
    for index in range(h-1, int(h/2), -1):
        if np.mean(sample_numpy[index, :, 0]) > threshold:
            bottom_index = index - 1
        else:
            break
    # Left to Right
    left_index = 0
    for index in range(int(w/2)):
        if np.mean(sample_numpy[:, index, 0]) > threshold:
            left_index = index + 1
        else:
            break
    # Right to Left
    right_index = w - 1
    for index in range(w - 1, int(w/2), -1):
        if np.mean(sample_numpy[:, index, 0]) > threshold:
            right_index = index - 1
        else:
            break
    # print(top_index,bottom_index+1, left_index,right_index+1)
    sample_numpy = sample_numpy[top_index:bottom_index+1, left_index:right_index+1]

    result_image = Image.fromarray(sample_numpy)
    if inverse:
        result_image = Image.fromarray(1 - sample_numpy)
    return result_image

if __name__ == '__main__':
    image = Image.open('D:/Projects/AMD_processed/01619194/test.png')
    image.show()
    image = cut(image)
    image.show()