import cv2
import numpy as np
import random

def follow_iou(gt_masks, region_mask, gt_objects_cls, ob_cls_for_search, pre_ious_matrix, remain_objects):
    '''

    :param gt_masks: np.array [image.shape[0], image.shape[1], objects_num]
    :param region_mask: array [image.shape[0], image.shape[1]]
    :param gt_objects_cls: array [objects_num]
    :param ob_cls_for_search: constant
    :param pre_ious_matrix: array[objects_num]  store the iou of each object of the ground truth,
                        just in case the agent changes of observed object.
    :param remain_objects: array [objects_num] 1 means still remain
    :return:
    '''
    objects_num = gt_objects_cls.shape[0]
    ious_for_ob = np.zeros([objects_num, 1])
    for k in range(objects_num):
        if gt_objects_cls[k] == ob_cls_for_search:
            if remain_objects[k] == 1:
                gt_mask = gt_masks[:, :, k]
                iou = calculate_iou(region_mask, gt_mask)
                ious_for_ob[k] = iou
            else:
                ious_for_ob[k] = -1
    max_result = max(ious_for_ob)
    ind = np.argmax(ious_for_ob)
    iou = pre_ious_matrix[ind]
    new_iou = max_result

    return iou, new_iou, ious_for_ob, ind


def calculate_iou(img_mask, gt_mask):
    '''
    :param img_mask: image.shape with visialbe area setting 1
    :param gt_mask: image.shape with gt_bbox area setting 1
    :return:
    iou(float) = cross_area/total_area
    '''
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j) / float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    '''
        :param img_mask: image.shape with visialbe area setting 1
        :param gt_mask: image.shape with gt_bbox area setting 1
        :return:
        overlap(float) = cross_area/gt_bbox_area
        '''
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j) / float(i))
    return overlap


def get_new_region_mask(action, scale_subregion, scale_shift, offset, pre_region_image, original_shape):
    '''
    :param action: 6 actions
    :param size_crop_image: []
    :param scale_subregion: constant 3/4
    :param scale_shift: constant 1/3
    :param offset: [0,0]: left corner position
    :param pre_region_image: pre hircarcy area
    :param original_shape:original image_shape
    :return:
    new_region_image
    new_region_mask
    '''
    new_region_mask = np.zeros(original_shape)
    size_mask = [pre_region_image.shape[0], pre_region_image.shape[1]]
    size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
    offset_aux = (1, 0)
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, size_mask[1] * scale_shift)
        offset = (offset[0], offset[1] + size_mask[1] * scale_shift)
    elif action == 3:
        offset_aux = (size_mask[0] * scale_shift, 0)
        offset = (offset[0] + size_mask[0] * scale_shift, offset[1])
    elif action == 4:
        offset_aux = (size_mask[0] * scale_shift,
                      size_mask[1] * scale_shift)
        offset = (offset[0] + size_mask[0] * scale_shift,
                  offset[1] + size_mask[1] * scale_shift)
    elif action == 5:
        offset_aux = (size_mask[0] * scale_shift / 2,
                      size_mask[1] * scale_shift / 2)
        offset = (offset[0] + size_mask[0] * scale_shift / 2,
                  offset[1] + size_mask[1] * scale_shift / 2)
    new_region_image = pre_region_image[offset_aux[0]:int(offset_aux[0] + size_mask[0]),
                   int(offset_aux[1]):int(offset_aux[1] + size_mask[1])]
    new_region_mask[int(offset[0]):int(offset[0]) + int(size_mask[0]), int(offset[1]):int(offset[1] + size_mask[1])] = 1

    return new_region_image, new_region_mask, offset


