""" Miscellaneous """

import cv2
import numpy as np
from scipy import ndimage


def pick_one_object_w_filled_holes(mask):
    """ return biggest object with filled holes
    """
    assert len(mask.shape) == 2, "len(mask.shape) != 2"
    #  fix some broken connection
    #  kernel = np.ones((10, 10), np.uint8)  # 20 is a magic number
    #  mask = cv2.dilate(mask, kernel, iterations=1)
    #  mask = cv2.erode(mask, kernel, iterations=1)
    #  fill the hole (ndimage.binary_fill_holes() return a binary mask)
    fixed_mask = np.asarray(ndimage.binary_fill_holes(mask) * 255, np.uint8)
    fixed_mask = pick_largest_object(fixed_mask) * fixed_mask
    return fixed_mask


def get_bb_location(mask):
    """ return (x_st, x_end, y_st, y_end)
    """
    assert len(mask.shape) == 2, "len(mask.shape) != 2"
    bbox = np.argwhere(mask)
    (y_st, x_st), (y_end, x_end) = bbox.min(0), bbox.max(0)
    return (x_st, x_end, y_st, y_end)


def crop_img(cv_img, roi_mask, pad_value=0, square=True, radius=None):
    """ return cropped image
    """
    assert cv_img.shape[:2] == roi_mask.shape, "shape not match"
    (x_st, x_end, y_st, y_end) = get_bb_location(roi_mask)

    if square and radius is not None:
        y_offset = int((2 * radius - (y_st + y_end)) / 2)
        x_offset = int((2 * radius - (x_st + x_end)) / 2)

        cropped_cv_img = np.zeros((2 * radius, 2 * radius, 3), np.uint8) + \
            pad_value
        cropped_cv_img[y_st + y_offset:y_end + y_offset,
                       x_st + x_offset:x_end + x_offset, :] = \
            cv_img[y_st:y_end, x_st:x_end, :]
    else:
        height = y_end - y_st
        width = x_end - x_st

        cropped_cv_img = np.zeros((height, width, 3), np.uint8)
        cropped_cv_img = cv_img[y_st:y_end, x_st:x_end, :]
    return cropped_cv_img


def crop_img_w_offset(cv_img, roi_mask, pad_value=0, square=True, radius=None):
    """ return cropped image
    """
    assert cv_img.shape[:2] == roi_mask.shape, "shape not match"
    (x_st, x_end, y_st, y_end) = get_bb_location(roi_mask)

    if square and radius is not None:
        y_offset = int((2 * radius - (y_st + y_end)) / 2)
        x_offset = int((2 * radius - (x_st + x_end)) / 2)

        cropped_cv_img = np.zeros((2 * radius, 2 * radius, 3), np.uint8) + \
            pad_value
        cropped_cv_img[y_st + y_offset:y_end + y_offset,
                       x_st + x_offset:x_end + x_offset, :] = \
            cv_img[y_st:y_end, x_st:x_end, :]
        return cropped_cv_img, x_offset, y_offset
    else:
        height = y_end - y_st
        width = x_end - x_st

        cropped_cv_img = np.zeros((height, width, 3), np.uint8) + pad_value
        cropped_cv_img = cv_img[y_st:y_end, x_st:x_end, :]
        return cropped_cv_img
    

def pick_largest_object(mask):
    """ return object of largest area
    args
        mask: threshold image (0 or 255)
    """
    connectivity = 8
    out_of_ccpn = cv2.connectedComponentsWithStats(mask, connectivity,
                                                   cv2.CV_32S)
    #  num_labels = out_of_ccpn[0]
    labels_matrix = out_of_ccpn[1]
    stats = out_of_ccpn[2]
    index_max_area = np.argmax(stats[1:, 4])
    return labels_matrix == (index_max_area + 1)


def scale_radius(cv_img, ratio, check_resolution=False):
    """ scale image according to radius of ROI
    """
    #  Img should be sclae down, not scale up.
    if check_resolution and ratio > 1:
        print("Image resolution is not as good as we wish.")
        raise ValueError
    return cv2.resize(cv_img, (0, 0), fx=ratio, fy=ratio)


def vis_detection_result(img, mask, save_path=None):
    mask = cv2.resize(mask, img.shape[:2][::-1])
    edges = cv2.Canny(255*mask, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)
    img[edges == 255] = 255
    return img
