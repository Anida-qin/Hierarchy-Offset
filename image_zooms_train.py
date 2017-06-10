#coding=utf-8
import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io

import random
import argparse
import tensorflow as tf

#export CUDA_VISIBLE_DEVICES = 1

from dataset.img_helper import *
from dataset.PacalVoc_xml_annotation import *
from nets.image_zoom_model import *
from nets.reinforcement import *
from my_utils.process_region import *
from utils.visualization import *


def get_train_data(replay, category, point_to_replay, state, action, reward, new_state,sess):
    if point_to_replay[category] < (buffer_experience_replay - 1):
        point_to_replay[category] += 1
    else:
        point_to_replay[category] = 0
    h_aux = point_to_replay[category]
    h_aux = int(h_aux)
    replay[category][h_aux] = (state, action, reward, new_state)
    minibatch = random.sample(replay[category], batch_size)
    X_train = []
    y_train = []
    # we pick from the replay memory a sampled minibatch and generate the training samples
    for memory in minibatch:
        old_state, action, reward, new_state = memory
        old_state_vgg = np.expand_dims(old_state,axis=0)
        old_qval = sess.run(model.qval_net,feed_dict={state_pl:old_state_vgg})
        new_state = np.expand_dims(new_state, axis=0)
        newQ = sess.run(model.qval_net,feed_dict={state_pl:new_state})
        maxQ = np.max(newQ)
        y = old_qval
        y = np.squeeze(y)
        if action != 10:  # non-terminal state
            update = (reward + (gamma * maxQ))
        else:  # terminal state
            update = reward
        y[action - 1] = update  # target output
        X_train.append(old_state)
        y_train.append(y)
    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)

    return X_train,y_train




if __name__ == "__main__":
    ######## PATHS definition ########


    # path of pascal voc test
    path_voc_train = "/home/qs/DATA/VOC2007_trainval/"
    #path_model = "/home/qs/models/models_image_zooms/"
    path_to_save = '/home/qs/models/DQN/'
    name_model = 'qnetmodel_aeroplane.ckpt'
    #path_testing_folder = "/media/disk/Obresults/testing_visualizations"

    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3) / 4
    scale_shift = float(1) / (scale_subregion * 4)
    rpn_scale = float(1)/4
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 0

    buffer_experience_replay=1000
    batch_size = 100
    gamma = 0.9
    epochs = 50
    epoch_id = 0
    epsilon = 1

    h=np.zeros(20)
    replay = [[] for i in range(20)]
    reward = 0


    ######## LOAD IMAGE NAMES ########

    image_names = np.array([load_images_names_in_data_set('trainval', path_voc_train)])
    # models = get_array_of_q_networks_for_pascal("0", class_object)

    ######## LOAD Models ########
    image_pl = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_img')
    history_pl = tf.placeholder(tf.float32,[40,],name='act_history')
    labels_pl = tf.placeholder(tf.float32,[None,10], name='labels')
    state_pl = tf.placeholder(tf.float32,[None,25128],name='states')
    hps = HParams(lr_rate = 1e-6,mode='train')

    model = Image_zoom_net(hps,image_pl,state_pl,labels_pl,history_pl)
    model.build_graph()
    # print end_points
    # vars_list = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
    vas=tf.global_variables()
    vars_list = [v for v in tf.global_variables() if v.name.split('/')[0] == 'vgg_16']
    vars_to_store = [v for v in tf.global_variables() if v.name.split('/')[0] == 'Q_net']
    init_fn, feed_dict = slim.assign_from_checkpoint('/home/qs//models/vgg_16.ckpt', vars_list)
    saver = tf.train.Saver(vars_to_store,max_to_keep=100)

    with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': True})) as sess:
        #vas = slim.get_model_variables()
        sess.run(tf.global_variables_initializer())
        sess.run(init_fn, feed_dict)
        #saver.restore(sess,'/home/qs/Desktop/models//'+'qnetmodel.ckpt-21')
        #epoch_id = 21



    #images = get_all_images(image_names, path_voc_train)

        for i in range(epoch_id, epoch_id + epochs):
            for j in range(image_names.shape[1]):
                masked = 0
                not_finished = 1
                image_name = image_names[0][j]
                image = get_one_image(image_name, path_voc_train)
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name,path_voc_train)
                gt_masks = generate_bbox_pixel_from_per_annotation(annotation,image.shape)
                gt_objects_cls = annotation[:,0]
                objects_num = gt_objects_cls.shape[0]
                region_mask = np.ones([image.shape[0],image.shape[1]])
                remain_objects = np.ones(objects_num)

                ###objects for loop####
                for k in range(objects_num):
                    # Init visualization
                    # background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
                    # draw = ImageDraw.Draw(background)

                    if gt_objects_cls[k] == class_object:
                        gt_mask = gt_masks[:,:,k]
                        step = 0
                        new_iou = 0
                        pre_matrix = np.zeros(objects_num)
                        region_image = image
                        offset = (0,0)
                        original_shape = [image.shape[0],image.shape[1]]
                        old_region_mask = region_mask
                        region_mask = np.ones([image.shape[0],image.shape[1]])

                    # when one aeroplane is marked other objects compared old region_mask(bbox for pre aero)
                    # shouldn't be used to train. otherwise one bbox may have two objects.
                        if masked == 1:
                            for p in range(objects_num):
                                overlap = calculate_overlapping(old_region_mask,gt_masks[:,:,p])
                                if overlap > 0.6:
                                    remain_objects[p] = 0

                        if np.count_nonzero(remain_objects) == 0:
                            not_finished = 0

                        iou, new_iou, pre_matrix, ind_max_iou = follow_iou(gt_masks,region_mask,gt_objects_cls,
                                                                ob_cls_for_search=class_object,
                                                                pre_ious_matrix=pre_matrix,
                                                                remain_objects=remain_objects)
                        new_iou = iou
                        gt_mask = gt_masks[:,:,ind_max_iou]
                        history_vector = np.zeros(40)
                        #image_vgg = np.expand_dims(tf.image.resize_images(image,[224,224]),axis=0)
                        image_vgg = np.expand_dims(cv2.resize(image,(224,224)).astype(np.float32),axis=0)

                        state = sess.run(model.state_net,feed_dict={image_pl:image_vgg, history_pl:history_vector})

                        status = 1
                        action = 0
                        reward = 0

                        if step>number_of_steps:
                            step += 1


                        while(status == 1) & (step< number_of_steps) & not_finished:
                            category = int(gt_objects_cls[k] - 1)
                            state_in = np.expand_dims(state,axis=0)
                            qval_out = sess.run([model.qval_net],feed_dict={history_pl:history_vector,
                                                                            state_pl:state_in})
                            qval = np.squeeze(qval_out)
                        # background = draw_sequences(i, k, step, action, draw, region_image, background,
                        #                             path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                        #                             bool_draw)
                            step += 1
                            if (i<100) & (new_iou > 0.5):
                                action = 10
                            elif random.random()< epsilon:
                                action = np.random.randint(1,10)
                            else:
                                action = np.argmax(qval)

                            if action == 10:
                                iou, new_iou, pre_matrix, ind_max_iou = follow_iou(gt_masks, region_mask, gt_objects_cls,
                                                                               ob_cls_for_search=class_object,
                                                                               pre_ious_matrix=pre_matrix,
                                                                               remain_objects=remain_objects)
                                gt_mask = gt_masks[:, :, ind_max_iou]
                                reward = get_reward_trigger(new_iou)
                                step += 1
                            else:
                                print region_image.shape
                                if region_image.shape[0]<=0:
                                    break
                                print action
                                if action == None:
                                    break
                                region_image, region_mask, offset = \
                                    get_new_region_mask(action, scale_subregion, scale_shift, rpn_scale, offset,
                                                        region_image, original_shape, image)
                                iou, new_iou, pre_matrix, ind_max_iou = follow_iou(gt_masks, region_mask, gt_objects_cls,
                                                                               ob_cls_for_search=class_object,
                                                                               pre_ious_matrix=pre_matrix,
                                                                               remain_objects=remain_objects)

                                gt_mask = gt_masks[:, :, ind_max_iou]
                                reward = get_reward_sign(iou, new_iou)
                                iou = new_iou


                            history_vector = update_history_vector_for_state(history_vector,action)
                            region_image_vgg = np.expand_dims(cv2.resize(region_image,(224,224)).astype(np.float32),axis=0)
                            new_state = sess.run(model.state_net,feed_dict={image_pl:region_image_vgg,history_pl:history_vector})

                            if len(replay[class_object-1])<(buffer_experience_replay):
                                replay[class_object-1].append([state,action,reward, new_state])
                            else:

                                X_train,y_train = get_train_data(replay,category,h,state,action,reward, new_state,sess)
                                global_step, lr_rate, cost,train_op = sess.run([model.global_step,model.lr_rate,model.cost,model.train_op],
                                                         feed_dict={image_pl:image_vgg,
                                                                    state_pl:X_train,
                                                                    labels_pl:y_train,
                                                                    history_pl:history_vector})
                                print global_step
                                print('%s epoch for %s image with %s steps %s lr: cost : %s\n'% (i,j,step,cost,lr_rate))

                                #hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

                                state = new_state
                            if action == 10:
                                status = 0
                                masked = 1
                                image = mask_image_with_mean_background(gt_mask, image)
                            else:
                                masked = 0
                        remain_objects[ind_max_iou] = 0
                        #saver.save(sess, '/home/qs/Desktop/models/' + 'qnetmodeltest.ckpt', global_step=i + 1)

            if epsilon>0.1:
                epsilon -= 0.1

            if(i%2 == 0):

                saver.save(sess,path_to_save+name_model,global_step=i+1)
            else:
                pass
















