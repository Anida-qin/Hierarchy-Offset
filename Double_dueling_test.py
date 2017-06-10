# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from dataset.img_helper import *
from dataset.PacalVoc_xml_annotation import *
from nets.Double_dueling_net import *
from nets.reinforcement import *
from my_utils.process_region import *
from my_utils.drawing import *
from my_utils.metrics import *


if __name__ == "__main__":

    ######## PATHS definition ########
    path_voc_test = "/home/qs/DATA/VOC2007_test/"
    path_to_save = '/home/qs/Double_Duel_1000_results/'
    name_model =  'qnetmodel_aero.ckpt'

    ######## PARAMETERS ########
    # Class category of PASCAL that the RL agent will be searching
    object_for_search = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3) / 4
    scale_shift = float(1) / (scale_subregion * 4)
    rpn_scale = float(1) / 4
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 0
    # Double DQN
    tau = 0.1

    # buffer params
    buffer_experience_replay=5000
    his_num = 40
    batch_size = 100
    gamma = 0.9
    epochs = 50
    epochs_id = 0
    epsilon = 1.0

    point = 0
    buffer = []
    reward = 0
    hist_per_ob = []

    ######## LOAD IMAGE NAMES ########

    #image_names = np.array([load_images_names_in_data_set('trainval', path_voc_train)])
    image_names = np.array([load_images_names_in_data_set('aeroplane_test', path_voc_test)])
    labels = load_images_labels_in_data_set('aeroplane_test', path_voc_test)


    ######## Build Graph ########
    tf.reset_default_graph()
    vgg = vgg_net()
    with tf.variable_scope('mainQN'):
        mainQN = Q_net('mainQN')
    with tf.variable_scope('targetQN'):
        targetQN = Q_net('targetQN')
    writer = tf.summary.FileWriter('/home/qs/tensorboard/',tf.get_default_graph())
    writer.close()

    ######### Load Vgg16 Model ############
    vas=tf.trainable_variables()
    vars_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'vgg_16']
    vars_to_store = [v for v in tf.trainable_variables() if v.name.split('/')[1] == 'Q_net']
    init_fn, feed_dict = slim.assign_from_checkpoint('/home/qs/models/vgg_16.ckpt', vars_list)
    saver = tf.train.Saver(vars_to_store)

    ### metric ####
    f = open('precision_recall_all', 'write')
    f1 = open('precision_recall_final', 'write')
    f2 = open('iou', 'write')
    f3 = open('TP_FP_FN', 'write')
    f4 = open('precision_recall_56', 'write')
    sum_p = [0] * 100
    sum_r = [0] * 100
    count = [0] * 100
    precision_recall_list = []
    TP_FP_FN_list = []


    ######## Start Session ###############
    with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': True})) as sess:
        ###### Load model ######
        sess.run(tf.global_variables_initializer())
        sess.run(init_fn, feed_dict)
        saver.restore(sess, '/home/qs/models/Double_Duel_58/' + 'qnetmodel_aero.ckpt-9')
        t = get_weights()
        a = sess.run(t)

        # metric all
        confidence_list_all = []

        for j in range(np.size(image_names)):
            if labels[j] == '1':
                ##### load image and labels #####
                image_name = image_names[0][j]
                image = get_one_image(image_name, path_voc_test)
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_test)
                gt_objects_cls = annotation[:, 0]
                objects_num = gt_objects_cls.shape[0]
                image_for_search = image.copy()
                image_to_draw = image.copy()
                ##### Define masks #####
                gt_masks = generate_bbox_pixel_from_per_annotation(annotation, image.shape)
                region_mask = np.ones([image.shape[0], image.shape[1]])
                remain_objects = np.ones(objects_num)
                masked = 0  ### remove untrained gt
                not_finished = 1
                original_shape = (image.shape[0], image.shape[1])
                offset = [0.0,0.0]
                # absolute status is a boolean we indicate if the agent will continue
                # searching object or not. If the first object already covers the whole
                # image, we can put it at 0 so we do not further search there
                absolute_status = 1
                action = 0
                step = 0
                qval = 0
                ##### metrics #####
                confidence = []
                iou_list = []

                while (step < number_of_steps) and (absolute_status == 1):
                    # init 1 step
                    step_aux = 0
                    region_image = image_for_search
                    region_mask = np.ones([image.shape[0], image.shape[1]])
                    status = 1
                    history_vector = np.zeros(his_num)
                    image_vgg = np.expand_dims(cv2.resize(region_image, (224, 224)).astype(np.float32), axis=0)

                    image_fea = sess.run(vgg.imgfea, feed_dict={vgg.image:image_vgg})
                    state = np.concatenate((np.reshape(image_fea, [-1]), history_vector), axis=0)
                    draw_new = True

                    while (status == 1) & (step < number_of_steps):
                        step += 1
                        state_in = np.expand_dims(state, axis=0)
                        action = sess.run(mainQN.predict, feed_dict={mainQN.state_input: state_in})+1
                        qval = sess.run(targetQN.qval,feed_dict={targetQN.state_input:state_in})
                        # print qval

                        reward = qval[0][action-1]
                        # draw_sequence(step, action, reward, image_name, offset, region_image, path_to_save, image)

                        # metric
                        confidence.append(qval[0][9])

                        if action != 10:
                            ###################
                            image_to_draw = image.copy()
                            region_image, region_mask, offset = \
                                get_new_region_mask(action, scale_subregion, scale_shift, rpn_scale, offset,
                                                    region_image, original_shape, image_for_search)

                            draw_sequence(step, action, reward, image_name, offset,
                                          region_image, path_to_save, image_to_draw, draw_new=False, color=(0, 255, 0))
                        if action == 10:
                            image_to_draw = image.copy()

                            draw_new = draw_sequence(step, action, reward, image_name, offset,
                                                     region_image, path_to_save, image, draw_new=True, color=(0, 0, 255))
                            # cls,flag = caculate_iou_cls(region_mask, gt_masks, flag, threhold=0.8)
                            # cls_list.append(cls)
                            offset = (0, 0)
                            status = 0
                            # if step == 1 or step_aux == 1:
                            #     absolute_status = 0
                            hist_per_ob.append(step)
                            if step == 1:
                                absolute_status = 0
                            if only_first_object == 1:
                                absolute_status = 0

                            # if step_aux == 1 and step != 1:
                            #     absolute_status = 0
                            image_for_search = mask_image_with_mean_background(region_mask, image_for_search)
                            region_image = image_for_search
                            cv2.imwrite('imageforsearch.png', image_for_search)
                            cv2.imwrite('image.png', image)

                        ### metric
                        iou_list_per = calculate_iou_all(img_mask=region_mask, gt_masks=gt_masks)
                        iou_list.append(iou_list_per)

                        # cls, flag = caculate_iou_cls(region_mask, gt_masks, flag, threhold=0.5)
                        # cls_list.append(cls)

                        history_vector = update_history_vector_for_state(history_vector, action)
                        region_vgg = np.expand_dims(cv2.resize(region_image, (224, 224)).astype(np.float32), axis=0)

                        image_fea = sess.run(vgg.imgfea, feed_dict={vgg.image: region_vgg})
                        new_state = np.concatenate((np.reshape(image_fea, [-1]), history_vector), axis=0)
                        state = new_state

                        # draw_sequence(step, action, reward, image_name, offset, region_image, path_to_save, image)


                        # evaluation
                f2.write(str(len(iou_list)))
                f2.write('\n')
                print iou_list
                # precision_recall_c_new = precision_recall_curve_new(iou_list,gt_masks,confidence)
                # precison_recall_c = precision_recall_curve(cls_list,gt_masks,flag,confidence)
                # 2017/0504/
                precision_recall_c_new, TP_FP_FN_per = precision_recall_curve_threhold_new_all(iou_list, gt_masks,
                                                                                               confidence)
                print precision_recall_c_new
                f.write(str(precision_recall_c_new))
                f.write("\n")

                for i in range(len(precision_recall_c_new)):
                    # print precision_recall_c_new[i][0]
                    # print precision_recall_c_new[i][1]
                    if precision_recall_c_new[i][0] != -1 and precision_recall_c_new[i][1] != -1:
                        sum_p[i] += precision_recall_c_new[i][0]
                        sum_r[i] += precision_recall_c_new[i][1]
                        count[i] += 1

                precision_recall_list.append(precision_recall_c_new)
                print len(precision_recall_list)
                ### new for all ###
                TP_FP_FN_list.append(TP_FP_FN_per)
                f3.write(str(TP_FP_FN_per))
                f3.write('\n')

        for i in range(100):
            if count[i] != 0:
                f1.write(str([float(sum_p[i]) / count[i], float(sum_r[i]) / count[i]]))
                f1.write("\n")
        f.close()
        f1.close()
        f2.close()

        precision_list = []
        recall_list = []
        precision_recall_for_56 = []
        for j in range(100):
            TP = 0
            FP = 0
            FN = 0
            for i in range(len(TP_FP_FN_list)):
                TP += TP_FP_FN_list[i][j][0]
                FP += TP_FP_FN_list[i][j][1]
                FN += TP_FP_FN_list[i][j][2]
            print (TP, FP, FN)
            precision = float(TP) / (TP + FP + np.finfo(np.float64).eps)
            precision_list.append(precision)
            recall = float(TP) / (TP + FN)
            recall_list.append(recall)
            precision_recall_for_56.append((precision, recall))
            f4.write(str((precision, recall)))
            f4.write('\n')
        AP = voc_ap(np.array(recall_list),np.array(precision_list),True)
        print 'AP'
        print AP
        AP2 = voc_ap(recall_list,precision_list,False)
        print 'AP2'
        print AP2
        f3.close()
        f4.close()

        hist_graph = [0]*10
        for i in range(len(hist_per_ob)):
            if  hist_per_ob[i]==1:
                hist_graph[0]+=1
            if  hist_per_ob[i]==2:
                hist_graph[1]+=1
            if hist_per_ob[i]==3:
                hist_graph[2]+=1
            if  hist_per_ob[i]==4:
                hist_graph[3]+=1
            if  hist_per_ob[i]==5:
                hist_graph[4]+=1
            if hist_per_ob[i]==6:
                hist_graph[5]+=1
            if  hist_per_ob[i]==7:
                hist_graph[6]+=1
            if  hist_per_ob[i]==8:
                hist_graph[7]+=1
            if hist_per_ob[i]==9:
                hist_graph[8]+=1
            if hist_per_ob[i]==10:
                hist_graph[9]+=1
        print hist_graph



