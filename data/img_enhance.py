""" Image Enhancement """

import cv2
import numpy as np


def fit(img, c_x, c_y, radius, pad_value=0):
    """ no enhancement, delete unwanted noise according to ROI
    It is duplicate with delete_noise(), please use delete_noise(). 
    """
    # create ROI mask
    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)

    pre_img = img * roi_mask_ch3 + pad_value * (1 - roi_mask_ch3)
    return pre_img


def delete_noise(img, roi_mask_ch3, pad_value=0):
    """ no enhancement, delete unwanted noise according to ROI
    """
    pre_img = img * roi_mask_ch3 + pad_value * (1 - roi_mask_ch3)
    return pre_img


def unsharp_masking(img, c_x, c_y, radius, strength_factor=30):
    """ enhancement method based on min-pooling
    """
    # create ROI mask
    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)

    pad_value = 128
    gblur_cv_img = cv2.GaussianBlur(img, (0, 0), radius / strength_factor)
    pre_img = cv2.addWeighted(img, 4, gblur_cv_img, -4, 128)
    pre_img = pre_img * roi_mask_ch3 + pad_value * (1 - roi_mask_ch3)
    return pre_img


def unsharp_masking_w_bilateral(img, c_x, c_y, radius):
    """ enhancement method based on min-pooling
    """
    # create ROI mask
    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)

    pad_value = 128
    gblur_cv_img = cv2.bilateralFilter(img, int(radius / 10.24), 75, 75)
    pre_img = cv2.addWeighted(img, 4, gblur_cv_img, -4, 128)
    pre_img = pre_img * roi_mask_ch3 + pad_value * (1 - roi_mask_ch3)
    return pre_img


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=10., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X


def global_contrast_normalize_3ch(img):
    img_gcn = np.zeros(img.shape)
    img_gcn[:, 0] = np.reshape(global_contrast_normalize(
        np.reshape(img[:, 0].flatten(), (1, -1)), use_std=True),
        img.shape[:1])
    img_gcn[:, 1] = np.reshape(global_contrast_normalize(
        np.reshape(img[:, 1].flatten(), (1, -1)), use_std=True),
        img.shape[:1])
    img_gcn[:, 2] = np.reshape(global_contrast_normalize(
        np.reshape(img[:, 2].flatten(), (1, -1)), use_std=True),
        img.shape[:1])

    def normalization_0255(img):
        """Normalize the img from 0 to 255"""
        img = img.astype(np.float)
        img = img - img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        return img
    print(img_gcn[:, 0].min(), img_gcn[:, 0].max(), img_gcn[:, 0].var(), img_gcn[:, 0].mean())
    # Normalize channels to between -1.0 and +1.0
    img_gcn[:, 0] = img_gcn[:, 0] / abs(img_gcn[:, 0]).max()
    img_gcn[:, 1] = img_gcn[:, 1] / abs(img_gcn[:, 1]).max()
    img_gcn[:, 2] = img_gcn[:, 2] / abs(img_gcn[:, 2]).max()
    print(img_gcn[:, 0].min(), img_gcn[:, 0].max(), img_gcn[:, 0].var(), img_gcn[:, 0].mean())
    # Normalize channels to between 0 and +255.0
    img_gcn = normalization_0255(img_gcn)
    print(img_gcn[:, 0].min(), img_gcn[:, 0].max(), img_gcn[:, 0].var(), img_gcn[:, 0].mean())
    return img_gcn


# convert RGB img in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

# histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, img is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

# My pre processing (use for both training and testing!)
def gray_normalization(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs
