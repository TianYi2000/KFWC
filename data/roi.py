"""ROI detection"""
from __future__ import division
import sys
import time
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

# logging
from logging import getLogger, NullHandler
import cv2
import numpy as np
from skimage import feature
from skimage.measure import ransac, CircleModel
from misc import get_bb_location
from misc import pick_one_object_w_filled_holes


LOGGER = getLogger(__name__)
LOGGER.addHandler(NullHandler())


def pre_roi_detection(img, img_path="None"):
    """
    input:
        img: numpy (BGR)
        img_path: str
    """
    assert len(img.shape) == 3, "len(img.shape) != 3"
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img[gray_img>=np.percentile(gray_img, [99])] = 0   # np.percentile()求第99%分位的数值，把大数值归0
    blur_gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)   # 高斯滤波，高斯矩阵(5, 5)，标准差为0

    threshold = blur_gray_img.mean()
    blur_gray_img_50l = blur_gray_img * (blur_gray_img < threshold)

    # global thresholding
    _, th1 = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    #  Otsu's thresholding
    _, th2 = cv2.threshold(blur_gray_img_50l, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_total = th1 | th2

    try:
      # fill holes
      roi_mask = pick_one_object_w_filled_holes(th_total)

      # check two-stages thresholding is successful or not
      heigh, width = gray_img.shape
      area = width * heigh
      percentage = cv2.countNonZero(roi_mask) / area
      if percentage > 0.95:  # magic number
          LOGGER.warn('Two-stages thresholding fail for %s', img_path)

          # use global thresholding only
          roi_mask = pick_one_object_w_filled_holes(th1)
    except:
      h, w = gray_img.shape[:2]
      radius = int(h/2.)
      c_y = int(h/2.)
      c_x = int(w/2.)
      roi_mask = np.zeros(gray_img.shape[:2], dtype=np.uint8)
      cv2.circle(roi_mask, (c_x, c_y), radius, (255), -1, 8, 0)
    return roi_mask


def post_roi_detection(gray_img):
    """
    input:
        gray_img: numpy (one channel)
        img_path: str
    """
    assert len(gray_img.shape) == 2, "len(gray_img.shape) !=2"
    edges = feature.canny(gray_img, sigma=1)
    points = np.array(np.nonzero(edges)).T

    model_robust, inliers = ransac(points, CircleModel, min_samples=3,
                                   residual_threshold=1, max_trials=300)
    c_y, c_x, radius = model_robust.params
    return c_y, c_x, radius, points, inliers


def roi_detection(img, img_path="None"):
    """
    input:
        img: numpy (BGR)
        img_path: str
    output:
        c_y: int
        c_x: int
        radius: int
        y_st: int
        y_end: int
    """
    assert len(img.shape) == 3, "len(img.shape) != 3"

    #  t_start = time.time()
    # ROI detection
    pre_roi_mask = pre_roi_detection(img, img_path)
    
    c_y, c_x, radius, _, _ = post_roi_detection(pre_roi_mask)

    # check post_roi_detection
    h, w = img.shape[:2]
    if radius > w:
        LOGGER.warn('Post-ROI detection fail for %s', img_path)

        # use green channel to replace pre-ROI mask
        c_y, c_x, radius, _, _ = post_roi_detection(img[:, :, 1])

        # check post_roi_detection again
        if radius > w:
          radius = h/2.
          c_y = h/2.
          c_x = w/2.

    # get bounding box location
    _, _, y_st, y_end = get_bb_location(pre_roi_mask)
    #  t_end = time.time()
    #  print "It cost %f sec" % (t_end - t_start)
    
    return int(c_y), int(c_x), int(radius), y_st, y_end
