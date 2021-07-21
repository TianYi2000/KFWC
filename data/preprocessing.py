""" Preprocessing """

from __future__ import division
from __future__ import absolute_import
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))

import time
from logging import getLogger, NullHandler
from math import ceil
import cv2
import numpy as np
from roi import roi_detection
from misc import scale_radius
from misc import crop_img, crop_img_w_offset
from img_enhance import fit
from img_enhance import delete_noise
from img_enhance import unsharp_masking
from img_enhance import unsharp_masking_w_bilateral
from img_enhance import global_contrast_normalize_3ch
from img_enhance import gray_normalization

# logging
LOGGER = getLogger(__name__)
LOGGER.addHandler(NullHandler())


def preprocessing_v1(img, mask_img=None, required_size=768, pre_flag=1,
                     factor=0.9, img_path="None",debug=False):
    """ preprocessing
    input
        factor (float): for boundary effect caused by unsharp masking
    """
    # inital setting
    if pre_flag == 1:
        required_r = ceil(required_size / 2 / factor)
    else:
        required_r = ceil(required_size / 2)

    # ROI detection
    t_start = time.time()
    scale_len = 400.
    w, h = img.shape[:2]
    max_len = max(w, h)
    scale = scale_len / max_len
    resized_img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
    c_y, c_x, radius, y_st, y_end = roi_detection(resized_img, img_path)
    c_y =  int(c_y / scale)
    c_x = int(c_x / scale)
    radius = int(radius / scale)
    y_st = int(y_st / scale)
    y_end = int(y_end / scale)
    t_end = time.time()
    print("ROI: It cost %f sec" % (t_end - t_start))
    
    t_start = time.time()
    # scale image to the requirement
    ratio = required_r / radius
    if ratio < 1:
        img = scale_radius(img, ratio)
        if mask_img is not None:
            mask_img = scale_radius(mask_img, ratio)

        # update parameters based on ratio
        c_x = int(ratio * c_x)
        c_y = int(ratio * c_y)
        radius = int(ratio * radius)
    else:
        ratio = 1

    # check the updated parameters are correct
    if debug:
        t_cy, t_cx, t_r, _, _ = roi_detection(img)
        assert abs(t_cx - c_x) < 5
        assert abs(t_cy - c_y) < 5
        assert abs(t_r - radius) < 5

    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)

    # image enhancement based on ROI
    if pre_flag == 0:
        pad_value = 0
        cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
        enhanced_cv_img = fit(img, c_x, c_y, radius)

        # fix boundary effect for none circle case
        interval = 0
    elif pre_flag == 1:
        pad_value = 128
        radius = int(factor * radius)  # fix boundary effect
        cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
        enhanced_cv_img = unsharp_masking(img, c_x, c_y, radius)

        # fix boundary effect for none circle case
        interval = 0.01 * (y_end - y_st)
    else:
        print("pre_flag == 0 -> Pre method 0 (fit)")
        print("pre_flag == 1 -> Pre method 1 (min-pooling)")
        print("The other values are not supported yet")
        raise ValueError

    y_st = int(y_st * ratio + interval)
    y_end = int(y_end * ratio - interval)
    roi_mask_ch3[y_end:, :, 0] = 0
    roi_mask_ch3[:y_st, :, 0] = 0

    cropped_pre_cv_img = crop_img(enhanced_cv_img, roi_mask_ch3[:, :, 0],
                                  pad_value=pad_value, square=True,
                                  radius=radius)
    pre_img = cv2.resize(cropped_pre_cv_img, (required_size, required_size))
    t_end = time.time()
    print("Pre: It cost %f sec" % (t_end - t_start))
    if mask_img is not None:
        mask_img = crop_img(mask_img, roi_mask_ch3[:, :, 0],
                                      pad_value=pad_value, square=True,
                                      radius=radius)
        mask_img = cv2.resize(mask_img, (required_size, required_size))
    if abs(cropped_pre_cv_img.shape[0] - required_size) > 5:
        LOGGER.warn('Resize image for too many pixels: %s', img_path)
        print(cropped_pre_cv_img.shape[0], required_size)
    if mask_img is not None:
        return pre_img, mask_img
    else:
        return pre_img
    
    
def classification_preprocessing(img, pre_option, img_path="None",
                                 debug=False):
    """ preprocessing
    input
        factor (float): for boundary effect caused by unsharp masking
        required_size: the size of diameter of the ROI of the fundus image
    Note
        required_size is meaningful only if crop_flag is True
    """
    crop_flag = pre_option['crop_flag']
    factor = pre_option['factor']
    pre_flag = pre_option['pre_flag']
    strength_factor = pre_option['strength_factor']

    # ROI detection.
    PRE_DEFINE_SIZE = 400.
    h, w = img.shape[:2]
    max_h_w = max(h, w)
    scale = PRE_DEFINE_SIZE / max_h_w
    resized_img = cv2.resize(img, (0,0), fx=scale, fy=scale) 

    c_y, c_x, radius, y_st, y_end = roi_detection(resized_img, img_path)
    
    # Adjust the parameters due to the resizing.
    c_y = int(c_y / scale)
    c_x = int(c_x / scale)
    radius = int(radius / scale)
    y_st = int(y_st / scale)
    y_end = int(y_end / scale)
        
    if pre_option['required_size'] is None:
        required_size = 2 * radius # keep the original image size
    else:
        required_size = pre_option['required_size']
    required_r = ceil(required_size / 2 / factor)

    # Scale image to the requirement.
    ratio = required_r / radius
    if ratio < 1:
        img = scale_radius(img, ratio)

        # Update parameters based on ratio.
        c_x = int(c_x * ratio)
        c_y = int(c_y * ratio)
        y_st = int(y_st * ratio)
        y_end = int(y_end * ratio)
        radius = int(radius * ratio)
    else:
        ratio = 1

    # Check the updated parameters are correct.
    if debug:
        t_cy, t_cx, t_r, _, _ = roi_detection(img)
        assert abs(t_cx - c_x) <= 10
        assert abs(t_cy - c_y) <= 10
        assert abs(t_r - radius) <= 10

    # Create ROI mask.
    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
    roi_mask_ch3[:y_st, :, :] = 0
    roi_mask_ch3[y_end:, :, :] = 0

    # Crop image based on ROI pre_mask_ch1.
    pad_value = 0
    roi_img = delete_noise(img, roi_mask_ch3, pad_value=pad_value)
    if crop_flag:
        roi_img, x_offset, y_offset = crop_img_w_offset(roi_img.copy(),
            roi_mask_ch3[:, :, 0], pad_value=pad_value, square=True,
            radius=radius)

    if pre_flag >= 1:
        if crop_flag:
            # Update parameters according to roi_img
            h, w = roi_img.shape[:2]

            y_st += (radius - c_y)
            delta_y_end = radius - (h - c_y) + (h - y_end)

            c_x += x_offset
            c_y += y_offset

            y_st += c_y - radius
            y_st = max(0, y_st)
            y_end = h - delta_y_end
            y_end += c_y - radius

        if pre_flag == 1:
            # image enhancement based on ROI
            pad_value = 128
            radius = int(factor * radius)  # fix boundary effect
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_img = unsharp_masking(roi_img, c_x, c_y, radius,
                strength_factor=(strength_factor*factor))
            if factor < 1.:
                # fix boundary effect for none circle case
                interval = 0.05 * (y_end - y_st)
                y_st = int(y_st + interval)
                y_end = int(y_end - interval)
            
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            if crop_flag:
                pre_img = crop_img(pre_img, pre_mask_ch3[:, :, 0],
                    pad_value=pad_value, square=True, radius=radius)
            else:
                pre_img = delete_noise(pre_img, pre_mask_ch3,
                    pad_value=pad_value)
        elif pre_flag == 2:
            # image enhancement based on ROI
            pad_value = 128
            radius = int(factor * radius)  # fix boundary effect
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_img = unsharp_masking_w_bilateral(roi_img, c_x, c_y, radius)
            if factor < 1.:
                # fix boundary effect for none circle case
                interval = 0.01 * (y_end - y_st)
                y_st = int(y_st + interval)
                y_end = int(y_end - interval)
            
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            if crop_flag:
                pre_img = crop_img(pre_img, pre_mask_ch3[:, :, 0],
                    pad_value=pad_value, square=True, radius=radius)
            else:
                pre_img = delete_noise(pre_img, pre_mask_ch3,
                    pad_value=pad_value)
        elif pre_flag == 3:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            #pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            
            t_img_yuv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            # equalize the histogram of the Y channel
            t_img_yuv[:,:,0] = clahe.apply(t_img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            t_img_yuv = cv2.cvtColor(t_img_yuv, cv2.COLOR_YUV2BGR)

            #temp_mask = roi_img[:, :, 0] == 0
            #temp_mask *= roi_img[:, :, 1] == 0
            #temp_mask *= roi_img[:, :, 2] == 0

            #pre_mask_ch3[temp_mask > 0, :] = [0, 0, 0]
            #pre_mask_ch1 = pre_mask_ch3[:, :, 0]

            #temp = roi_img[pre_mask_ch1 > 0, :]
            #temp_gcn = global_contrast_normalize_3ch(temp.copy())

            pre_img = np.zeros(roi_img.shape, dtype=np.uint8)
            pre_img = pre_mask_ch3 * t_img_yuv
            #pre_img[pre_mask_ch1 > 0, :] = temp_gcn
        elif pre_flag == 4:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            
            roi_img = cv2.bilateralFilter(roi_img, 9, 75, 75)
            t_img_yuv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))

            # equalize the histogram of the Y channel
            t_img_yuv[:,:,0] = clahe.apply(t_img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            t_img_rgb = cv2.cvtColor(t_img_yuv, cv2.COLOR_YUV2BGR)

            pre_img = np.zeros(roi_img.shape, dtype=np.uint8)
            pre_img = pre_mask_ch3 * t_img_rgb
        elif pre_flag == 5:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, 0] = 0
            pre_mask_ch3[y_end:, :, 0] = 0
            pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            
            temp = roi_img.transpose(2, 0, 1)
            temp = np.expand_dims(temp, axis=0)
            temp = gray_normalization(temp)
            temp = temp[0][0]
            temp *= 255
            
            pre_img = np.zeros(roi_img.shape[:2], dtype=np.uint8)
            pre_img = pre_mask_ch1 * temp
        else:
            print("pre_flag == 0 -> Pre method 0 (fit)")
            print("pre_flag == 1 -> Pre method 1 (min-pooling)")
            print("pre_flag == 2 -> Pre method 2 (min-pooling w/ bilateral)")
            print("pre_flag == 3 -> Pre method 3 (CLALE)")
            print("pre_flag == 4 -> Pre method 4 (CLALE and bilateral filtering)")
            print("pre_flag == 5 -> Pre method 5 (gray normalize)")
            print("The other values are not supported yet")
            raise ValueError
        
        if crop_flag and pre_option['required_size'] is not None:
            if abs(pre_img.shape[0] - required_size) > 5:
                LOGGER.warn('Resize image for too many pixels: %s', img_path)
                LOGGER.warn('ori size vs. required size: %s, %s',
                    str(pre_img.shape[0]), str(required_size))
            pre_img = cv2.resize(pre_img, (required_size, required_size))
        else:
            if pre_option['required_size'] is not None:
                LOGGER.warn('required_size is fully meaningful only if crop_flag is True')
        return {
            "pre_img": pre_img,
            "pre_mask": pre_mask_ch3[:, :, 0],
            "roi_img": roi_img, 
            "roi_mask": roi_mask_ch3[:, :, 0]}
    else:
        if crop_flag and pre_option['required_size'] is not None:
            if abs(roi_img.shape[0] - required_size) > 5:
                LOGGER.warn('Resize image for too many pixels: %s', img_path)
                LOGGER.warn('ori size vs. required size: %s, %s',
                    str(roi_img.shape[0]), str(required_size))
            roi_img = cv2.resize(roi_img, (required_size, required_size))
        else:
            if pre_option['required_size'] is not None:
                LOGGER.warn('required_size is meaningful only if crop_flag is True')
        return {
            "roi_img": roi_img, 
            "roi_mask": roi_mask_ch3[:, :, 0]}
    

def segmentation_preprocessing(img, pre_option, masks, img_path="None",
                               debug=False):
    """ preprocessing
    input
        factor (float): for boundary effect caused by unsharp masking
        required_size: the size of diameter of the ROI of the fundus image
    Note
        required_size is meaningful only if crop_flag is True
    """
    crop_flag = pre_option['crop_flag']
    factor = pre_option['factor']
    pre_flag = pre_option['pre_flag']
    strength_factor = pre_option['strength_factor']

    # ROI detection.
    PRE_DEFINE_SIZE = 400.
    h, w = img.shape[:2]
    max_h_w = max(h, w)
    scale = PRE_DEFINE_SIZE / max_h_w
    resized_img = cv2.resize(img, (0,0), fx=scale, fy=scale) 

    c_y, c_x, radius, y_st, y_end = roi_detection(resized_img, img_path)
    
    # Adjust the parameters due to the resizing.
    c_y = int(c_y / scale)
    c_x = int(c_x / scale)
    radius = int(radius / scale)
    y_st = int(y_st / scale)
    y_end = int(y_end / scale)
        
    if pre_option['required_size'] is None:
        required_size = 2 * radius # keep the original image size
    else:
        required_size = pre_option['required_size']
    required_r = ceil(required_size / 2 / factor)

    # scale image to the requirement
    ratio = required_r / radius
    if ratio < 1:
        img = scale_radius(img, ratio)

        # update parameters based on ratio
        c_x = int(c_x * ratio)
        c_y = int(c_y * ratio)
        y_st = int(y_st * ratio)
        y_end = int(y_end * ratio)
        radius = int(radius * ratio)
    else:
        ratio = 1

    # Check the updated parameters are correct.
    if debug:
        t_cy, t_cx, t_r, _, _ = roi_detection(img)
        assert abs(t_cx - c_x) <= 10
        assert abs(t_cy - c_y) <= 10
        assert abs(t_r - radius) <= 10


    roi_mask_ch3 = np.zeros(img.shape, dtype=np.uint8)
    
    # Crop image based on ROI mask.
    pad_value = 0
    cv2.circle(roi_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
    roi_mask_ch3[:y_st, :, :] = 0
    roi_mask_ch3[y_end:, :, :] = 0
    roi_img = delete_noise(img, roi_mask_ch3, pad_value=pad_value)
    if crop_flag: 
        roi_img, x_offset, y_offset = crop_img_w_offset(roi_img,
            roi_mask_ch3[:, :, 0], pad_value=pad_value, square=True,
            radius=radius)
    
    roi_seg_masks = {}
    for key in masks.keys():
        roi_seg_mask = masks[key]
        if pre_option['required_size'] is not None:
            roi_seg_mask = scale_radius(roi_seg_mask, ratio)
        if crop_flag: 
            roi_seg_mask = crop_img(roi_seg_mask, roi_mask_ch3[:, :, 0],
                pad_value=pad_value, square=True, radius=radius)
        roi_seg_masks[key] = roi_seg_mask
    
    if pre_flag >= 1:
        if crop_flag: 
            # Update parameters according to roi_img
            h, w = roi_img.shape[:2]

            y_st += (radius - c_y)
            delta_y_end = radius - (h - c_y) + (h - y_end)

            c_x += x_offset
            c_y += y_offset

            y_st += c_y - radius
            y_end = h - delta_y_end
            y_end += c_y - radius

        if pre_flag == 1:
            # image enhancement based on ROI
            pad_value = 128
            radius = int(factor * radius)  # fix boundary effect
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_img = unsharp_masking(roi_img, c_x, c_y, radius,
                strength_factor=(factor * strength_factor))

            if factor < 1.:
                # fix boundary effect for none circle case
                interval = 0.05 * (y_end - y_st)
                y_st = int(y_st + interval)
                y_end = int(y_end - interval)
            
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            if crop_flag:
                pre_img = crop_img(pre_img, pre_mask_ch3[:, :, 0],
                    pad_value=pad_value, square=True, radius=radius)
            else:
                pre_img = delete_noise(pre_img, pre_mask_ch3,
                    pad_value=pad_value)
        elif pre_flag == 2:
            # image enhancement based on ROI
            pad_value = 128
            radius = int(factor * radius)  # fix boundary effect
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_img = unsharp_masking_w_bilateral(roi_img, c_x, c_y, radius)

            if factor < 1.:
                # fix boundary effect for none circle case
                interval = 0.01 * (y_end - y_st)
                y_st = int(y_st + interval)
                y_end = int(y_end - interval)
            
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            if crop_flag:
                pre_img = crop_img(pre_img, pre_mask_ch3[:, :, 0],
                    pad_value=pad_value, square=True, radius=radius)
            else:
                pre_img = delete_noise(pre_img, pre_mask_ch3,
                    pad_value=pad_value)
        elif pre_flag == 3:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            #pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            
            t_img_yuv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            # equalize the histogram of the Y channel
            t_img_yuv[:,:,0] = clahe.apply(t_img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            t_img_yuv = cv2.cvtColor(t_img_yuv, cv2.COLOR_YUV2BGR)

            #temp_mask = roi_img[:, :, 0] == 0
            #temp_mask *= roi_img[:, :, 1] == 0
            #temp_mask *= roi_img[:, :, 2] == 0

            #pre_mask_ch3[temp_mask > 0, :] = [0, 0, 0]
            #pre_mask_ch1 = pre_mask_ch3[:, :, 0]

            #temp = roi_img[pre_mask_ch1 > 0, :]
            #temp_gcn = global_contrast_normalize_3ch(temp.copy())

            pre_img = np.zeros(roi_img.shape, dtype=np.uint8)
            pre_img = pre_mask_ch3 * t_img_yuv
            #pre_img[pre_mask_ch1 > 0, :] = temp_gcn
        elif pre_flag == 4:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, :] = 0
            pre_mask_ch3[y_end:, :, :] = 0
            
            roi_img = cv2.bilateralFilter(roi_img, 9, 75, 75)
            t_img_yuv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))

            # equalize the histogram of the Y channel
            t_img_yuv[:,:,0] = clahe.apply(t_img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            t_img_rgb = cv2.cvtColor(t_img_yuv, cv2.COLOR_YUV2BGR)

            pre_img = np.zeros(roi_img.shape, dtype=np.uint8)
            pre_img = pre_mask_ch3 * t_img_rgb
        elif pre_flag == 5:
            pre_mask_ch3 = np.zeros(roi_img.shape, dtype=np.uint8)
            cv2.circle(pre_mask_ch3, (c_x, c_y), radius, (1, 1, 1), -1, 8, 0)
            pre_mask_ch3[:y_st, :, 0] = 0
            pre_mask_ch3[y_end:, :, 0] = 0
            pre_mask_ch1 = pre_mask_ch3[:, :, 0]
            
            temp = roi_img.transpose(2, 0, 1)
            temp = np.expand_dims(temp, axis=0)
            temp = gray_normalization(temp)
            temp = temp[0][0]
            temp *= 255
            
            pre_img = np.zeros(roi_img.shape[:2], dtype=np.uint8)
            pre_img = pre_mask_ch1 * temp
        else:
            print("pre_flag == 0 -> Pre method 0 (fit)")
            print("pre_flag == 1 -> Pre method 1 (min-pooling)")
            print("pre_flag == 2 -> Pre method 2 (min-pooling w/ bilateral)")
            print("pre_flag == 3 -> Pre method 3 (CLALE)")
            print("pre_flag == 4 -> Pre method 4 (CLALE and bilateral filtering)")
            print("pre_flag == 5 -> Pre method 5 (gray normalize)")
            print("The other values are not supported yet")
            raise ValueError
            
        pre_seg_masks = {}
        for key in roi_seg_masks.keys():
            roi_seg_mask = roi_seg_masks[key]
            pre_seg_mask = pre_mask_ch3 * roi_seg_mask
            if crop_flag: 
                pad_value = 0
                pre_seg_mask = crop_img(pre_seg_mask, pre_mask_ch1,
                    pad_value=pad_value, square=True, radius=radius)
                if pre_option['required_size'] is not None:
                    pre_seg_mask = cv2.resize(pre_seg_mask,
                        (required_size, required_size))
            pre_seg_masks[key] = pre_seg_mask
        
        if crop_flag and pre_option['required_size'] is not None:
            if abs(pre_img.shape[0] - required_size) > 5:
                LOGGER.warn('Resize image for too many pixels: %s', img_path)
                LOGGER.warn('ori size vs. required size: %s, %s',
                    str(pre_img.shape[0]), str(required_size))
            pre_img = cv2.resize(pre_img, (required_size, required_size))
        else:
            if pre_option['required_size'] is not None:
                LOGGER.warn('required_size is fully meaningful only if crop_flag is True')
        return {
            "pre_img": pre_img,
            "pre_mask": pre_mask_ch3[:, :, 0],
            "roi_img": roi_img, 
            "roi_mask": roi_mask_ch3[:, :, 0],
            "roi_seg_masks": roi_seg_masks,
            "pre_seg_masks": pre_seg_masks}
    else:
        if crop_flag and pre_option['required_size'] is not None:
            if abs(roi_img.shape[0] - required_size) > 5:
                LOGGER.warn('Resize image for too many pixels: %s', img_path)
                LOGGER.warn('ori size vs. required size: %s, %s',
                    str(roi_img.shape[0]), str(required_size))
            roi_img = cv2.resize(roi_img, (required_size, required_size))
        else:
            if pre_option['required_size'] is not None:
                LOGGER.warn('required_size is meaningful only if crop_flag is True')
        return {
            "roi_img": roi_img, 
            "roi_mask": roi_mask_ch3[:, :, 0],
            "roi_seg_masks": roi_seg_masks}
