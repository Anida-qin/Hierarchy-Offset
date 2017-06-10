import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

reward_sign = 1
reward_trigger = 3
iou_threhold = 0.5
actions_history_num = 4
actions_num = 10

def get_reward_sign(iou, new_iou):
    if iou<new_iou:
        reward = reward_sign
    else:
        reward = -reward_sign
    return reward

def get_reward_trigger(new_iou):
    if new_iou>iou_threhold:
        reward = reward_trigger
    else:
        reward = -reward_trigger
    return reward

def update_history_vector_for_state(history_vector, action):
    action_vector = np.zeros(actions_num)
    action_vector[action-1] = 1
    actions_num_in_his_vec = np.size(np.nonzero(history_vector))
    update_history_vector = np.zeros(actions_num*actions_history_num)
    if actions_num_in_his_vec < actions_history_num:
        a_index = 0
        for i in range(actions_num_in_his_vec*actions_num, actions_num_in_his_vec*actions_num+actions_num):
            history_vector[i] = action_vector[a_index]
            a_index += 1
        return history_vector
    else:
        for j in range(0,(actions_history_num-1)*actions_num):
            update_history_vector[j] = history_vector[j+actions_num]
        a_index = 0
        for j in range((actions_history_num-1)*actions_num, actions_history_num*actions_num):
            update_history_vector[j] = action_vector[a_index]
            a_index += 1
        return update_history_vector


