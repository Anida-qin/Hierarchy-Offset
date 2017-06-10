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
from utils.util import *
from utils.drawing import *


if __name__ == "__main__":
    ######## PATHS definition ########


    # path of pascal voc test
    path_voc_test = "/media/disk/ObDataset/VOC2007_test/"
    path_model = "/media/disk/Obresults/models_image_zooms2/"
    path_to_save = "/media/disk/Obresults/testing_visualizations"


    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3) / 4
    scale_shift = float(1) / (scale_subregion * 4)
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 1

    buffer_experience_replay=1000
    batch_size = 100
    gamma = 0.9
    epochs = 50
    epoch_id = 0
    epsilon = 1

    h=np.zeros(20)
    replay = [[] for i in range(20)]
    reward = 0
    # evaluate
    precision_recall_list = []


    ######## LOAD IMAGE NAMES ########

    image_names = np.array([load_images_names_in_data_set('aeroplane_test', path_voc_test)])
    labels = load_images_labels_in_data_set('aeroplane_test', path_voc_test)
    # models = get_array_of_q_networks_for_pascal("0", class_object)

    ######## LOAD Models ########
    image_pl = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_img')
    history_pl = tf.placeholder(tf.float32,[24,],name='act_history')
    labels_pl = tf.placeholder(tf.float32,[None,6], name='labels')
    state_pl = tf.placeholder(tf.float32,[None,25112],name='states')
    hps = HParams(lr_rate = 1e-6,mode='test')

    model = Image_zoom_net(hps,image_pl,state_pl,labels_pl,history_pl)
    model.build_graph()
    # print end_points
    # vars_list = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
    vas=tf.global_variables()
    vars_list = [v for v in tf.global_variables() if v.name.split('/')[0] == 'vgg_16']
    vars_to_store = [v for v in tf.global_variables() if v.name.split('/')[0] == 'Q_net']
    init_fn, feed_dict = slim.assign_from_checkpoint('/home/qs/Desktop/models/vgg_16.ckpt', vars_list)
    saver = tf.train.Saver(vars_to_store,max_to_keep=30)

    f = open('cls_list.txt','write')

    with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': True})) as sess:
        #vas = slim.get_model_variables()
        sess.run(tf.global_variables_initializer())
        sess.run(init_fn, feed_dict)
        saver.restore(sess,'/home/qs/Desktop/models/image_zooms2/'+'qnetmodel.ckpt-49')
        #epoch_id = 21

        for j in range(image_names.shape[1]):
            if labels[j] == '1':
                masked = 0
                not_finished = 1
                image_name = image_names[0][j]
                image = get_one_image(image_name, path_voc_test)
                original_shape = [image.shape[0], image.shape[1]]
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_test)
                gt_masks = generate_bbox_pixel_from_per_annotation(annotation, image.shape)
                gt_objects_cls = annotation[:, 0]
                objects_num = gt_objects_cls.shape[0]
                region_mask = np.ones([image.shape[0], image.shape[1]])
                remain_objects = np.ones(objects_num)

                offset = (0, 0)
                # absolute status is a boolean we indicate if the agent will continue
                # searching object or not. If the first object already covers the whole
                # image, we can put it at 0 so we do not further search there
                absolute_status = 1
                action = 0
                step = 0
                qval = 0
                image_for_search = image
                region_mask = np.ones([image.shape[0], image.shape[1]])
                # evaluation
                flag = [0] * gt_masks.shape[2]
                cls_list = []

                while(step < number_of_steps) and (absolute_status == 1):
                    # init 1 step
                    region_image = image_for_search
                    status = 1
                    history_vector = np.zeros(24)
                    image_vgg = np.expand_dims(cv2.resize(region_image, (224, 224)).astype(np.float32), axis=0)
                    state = sess.run(model.state_net, feed_dict={image_pl: image_vgg, history_pl: history_vector})

                    while (status == 1) & (step < number_of_steps):
                        step += 1
                        state_in = np.expand_dims(state,axis=0)
                        qval = sess.run(model.qval_net,feed_dict={state_pl:state_in})
                        action = (np.argmax(qval)) + 1

                        if action != '6':
                            region_image, region_mask, offset = get_new_region_mask(action,scale_subregion,scale_shift,offset,
                                                                                            pre_region_image= region_image,
                                                                                            original_shape=original_shape)
                        if action == 6:
                            cls, flag = caculate_iou_cls(region_mask, gt_masks, flag, threhold=0.5)
                            cls_list.append(cls)
                            offset = (0, 0)
                            status = 0
                            if step == 1:
                                absolute_status = 0
                            if only_first_object == 1:
                                absolute_status = 0
                            image_for_search = mask_image_with_mean_background(region_mask, image_for_search)
                            region_image = image_for_search
                        history_vector = update_history_vector_for_state(history_vector, action)
                        region_vgg = np.expand_dims(cv2.resize(region_image,(224,224)).astype(np.float32),axis=0)
                        new_state = sess.run(model.state_net,feed_dict={history_pl:history_vector,image_pl:region_vgg})
                        state = new_state
                        draw_sequence(step, action, reward, image_name, offset, region_image, path_to_save, image)

                print cls_list
                f.write(str(cls_list))
                f.write("\n")


                # evaluation
                if len(cls_list) == 0:
                    precision_recall_list.append((0, 0))
                if len(cls_list) != 0:
                    per_metric = precision_recall(cls_list, gt_masks)
                    precision_recall_list.append(per_metric)
                    print per_metric

        fl = open('list_plane_05.txt', 'w')
        for i in precision_recall_list:
            strlist = str(i)
            fl.write(strlist)
            fl.write("\n")
        fl.close()
        f.close()  # background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
