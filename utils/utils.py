from __future__ import print_function, division, absolute_import
import math

import sklearn
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# from munch import munchify
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calc_kappa(a, b, labels):
    to_dict = lambda x: {k: [1 if y[i] > 0.5 else 0 for y in x] for i, k in enumerate(labels)}
    a_dict = to_dict(a)
    b_dict = to_dict(b)
    # print(a_dict)
    # print(b_dict)
    cohen_dict = {k: cohen_kappa_score(a_dict[k], b_dict[k]) for k in labels}
    for key in labels:
        if cohen_dict[key] != cohen_dict[key]:      # 消除nan值，nan != nan
            del cohen_dict[key]
    # print(cohen_dict)
    cohen_avg = np.mean(list(cohen_dict.values()))
    return cohen_avg


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):
    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 random_rotate=False, 
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_rotate = random_rotate
        
        tfs = []
        
#         if preserve_aspect_ratio:
#             tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
#         else:
#             height = int(self.input_size[1] / self.scale)
#             width = int(self.input_size[2] / self.scale)
#             tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
#             tfs.append(transforms.RandomResizedCrop(max(self.input_size),scale=(0.9,1.1)))
#         else:#不需要centercrop
#             tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())
                              
        if random_rotate:
            tfs.append(transforms.RandomRotation(180))
        
#         tfs.append(transforms.RandomApply(random_tfs,p=0.6))
        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
#         tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        # print("tensor!")
        tensor = self.tf(img)
        return tensor


def plot(cm, classes, normalize=False,
          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(target, pred, class_names, normalize=True, title='Normalized Confusion Matrix'):

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target, pred)
    np.set_printoptions(precision=2)

    # Plot confusion matrix
    plt.figure()
    plot(cnf_matrix, classes=class_names, normalize=normalize, title=title)

    plt.show()

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)

    return  { 'hamming_score' : np.mean(acc_list) ,
              'subset_accuracy' : sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None),
              'hamming_loss' : sklearn.metrics.hamming_loss(y_true, y_pred)}