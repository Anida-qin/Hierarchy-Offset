import tensorflow as tf
import numpy as np
from process_region import *

def env(action, history_vector, gt_mask, sess, model_vgg):
    if action == 6:
        iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                      array_classes_gt_objects, class_object,
                                                      last_matrix, available_objects)
        gt_mask = gt_masks[:, :, index]
        reward = get_reward_trigger(new_iou)
        background = draw_sequences(i, k, step, action, draw, region_image, background,
                                    path_testing_folder, iou, reward, gt_mask, region_mask,
                                    image_name, bool_draw)
        step += 1
        # movement action, we perform the crop of the corresponding subregion
    else:
        region_mask = np.zeros(original_shape)
        size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
        get_new_region_mask(action,)
        iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask,
                                                      array_classes_gt_objects, class_object,
                                                      last_matrix, available_objects)
        gt_mask = gt_masks[:, :, index]
        reward = get_reward_movement(iou, new_iou)
        iou = new_iou
    history_vector = update_history_vector(history_vector, action)
    new_state = get_state(region_image, history_vector, model_vgg)