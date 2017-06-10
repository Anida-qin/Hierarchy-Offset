import tensorflow as tf
import numpy as np
from dataset.img_helper import *
from dataset.PacalVoc_xml_annotation import *
from nets.Double_dueling_net import *
from nets.reinforcement import *
from my_utils.process_region import *
from utils.visualization import *


if __name__ == "__main__":

    ######## PATHS definition ########
    path_voc_train = "/home/qs/DATA/VOC2007_trainval/"
    path_to_save = '/home/qs/models/aeroplane/'
    name_model = 'qnetmodel_aeroplane.ckpt'

    ######## PARAMETERS ########
    # Class category of PASCAL that the RL agent will be searching
    object_for_search = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3) / 4
    scale_shift = float(1) / (scale_subregion * 4)
    rpn_scale = float(1) / 5
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 1
    # Double DQN
    tau = 0.1

    # buffer params
    buffer_experience_replay=5000
    his_num = 60
    batch_size = 300
    gamma = 0.9
    epochs = 80
    epochs_id = 0
    epsilon = 1.0

    point = 0
    buffer = []
    reward = 0


    ######## LOAD IMAGE NAMES ########

    image_names = np.array([load_images_names_in_data_set('trainval', path_voc_train)])


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
    saver = tf.train.Saver(vars_to_store,max_to_keep=500)
    targetOps = updateTargetGraph(tf.trainable_variables(),tau)



    ######## Start Session ###############
    with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': True})) as sess:
        ###### Load model ######
        sess.run(tf.global_variables_initializer())
        sess.run(init_fn, feed_dict)
        updateTarget(targetOps,sess)
        #saver.restore(sess, '/home/qs/models/Double_Duel/' + 'qnetmodel_aero.ckpt-15')

        for i in range(epochs_id, epochs_id + epochs):
            for j in range(np.size(image_names)):
                ##### load image and labels #####
                image_name = image_names[0][j]
                image = get_one_image(image_name, path_voc_train)
                annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_train)
                gt_objects_cls = annotation[:, 0]
                objects_num = gt_objects_cls.shape[0]
                image_for_search = image.copy()
                ##### Define masks #####
                gt_masks = generate_bbox_pixel_from_per_annotation(annotation, image.shape)
                region_mask = np.ones([image.shape[0], image.shape[1]])
                remain_objects = np.ones(objects_num)
                masked = 0   ### remove untrained gt
                not_finished = 1
                original_shape = (image.shape[0],image.shape[1])

                # Iterate through all the objects in the ground truth of an image
                for k in range(objects_num):
                    # We check whether the ground truth object is of the target class category
                    if gt_objects_cls[k] == object_for_search:
                        ##### Define masks #####
                        gt_mask = gt_masks[:, :, k]
                        step = 0
                        new_iou = 0.0
                        # this matrix stores the IoU of each object of the ground-truth, just in case
                        # the agent changes of observed object
                        pre_iou_matrix = np.zeros(objects_num)
                        ### original image size with masks object has already found ####
                        region_image = image_for_search
                        offset = (0, 0)
                        # size_mask = (image.shape[0], image.shape[1])
                        # original_shape = size_mask
                        old_region_mask = region_mask
                        region_mask = np.ones([image.shape[0], image.shape[1]])
                        # If the ground truth object is already masked by other already found masks, do not
                        # use it for training
                        if masked == 1:
                            for p in range(objects_num):
                                overlap = calculate_overlapping(old_region_mask, gt_masks[:, :, p])
                                if overlap > 0.60:
                                    remain_objects[p] = 0
                        # We check if there are still obejcts to be found
                        if np.count_nonzero(remain_objects) == 0:
                            not_finished = 0
                        # follow_iou function calculates at each time step which is the ground truth object
                        # that overlaps more with the visual region, so that we can calculate the rewards appropiately
                        ### ??????? what this for ##################
                        iou, new_iou, pre_iou_matrix, index = follow_iou(gt_masks, region_mask, gt_objects_cls,
                                                                      object_for_search, pre_iou_matrix, remain_objects)
                        new_iou = iou
                        gt_mask = gt_masks[:, :, index]
                        #############################################
                        # init of the history vector that indicates past actions (6 actions * 4 steps in the memory)
                        history_vector = np.zeros([his_num])
                        # computation of the initial state
                        region_image_vgg = np.expand_dims(cv2.resize(region_image,(224,224)).astype(np.float32),axis=0)
                        # 1*7*7*512
                        image_fea = sess.run(vgg.imgfea,feed_dict={vgg.image:region_image_vgg})
                        state = np.concatenate((np.reshape(image_fea,[-1]),history_vector),axis=0)
                        # status indicates whether the agent is still alive and has not triggered the terminal action
                        status = 1
                        action = 0
                        reward = 0
                        ##### search for one gt object #####
                        while (status == 1) & (step < number_of_steps) & not_finished:

                            state_in = np.expand_dims(state,axis=0)
                            qval = sess.run(targetQN.qval,feed_dict={targetQN.state_input:state_in})
                            #print qval
                            step += 1
                            # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
                            if (i < 100) & (new_iou > 0.5):
                                action = 10
                            # epsilon-greedy policy
                            elif random.random() < epsilon:
                                action = np.random.randint(1, 11)
                            else:
                                action = sess.run(mainQN.predict,feed_dict={mainQN.state_input:state_in}) + 1

                            ##### actions ######
                            # terminal action
                            if action == 10:
                                iou, new_iou, pre_iou_matrix, index = follow_iou(gt_masks, region_mask, gt_objects_cls,
                                                                                 object_for_search, pre_iou_matrix,
                                                                                 remain_objects)
                                # calculate reward
                                gt_mask = gt_masks[:,:,index]
                                reward = get_reward_trigger(new_iou)
                                step += 1
                            else:
                                region_image, region_mask, offset = \
                                    get_new_region_mask(action, scale_subregion, scale_shift, rpn_scale, offset,
                                                    region_image, original_shape, image_for_search)
                                #print offset

                                iou, new_iou, pre_iou_matrix, index = follow_iou(gt_masks, region_mask, gt_objects_cls,
                                                                                 object_for_search, pre_iou_matrix,
                                                                                 remain_objects)
                                # calculate reward
                                gt_mask = gt_masks[:,:,index]
                                reward = get_reward_sign(iou,new_iou)
                                iou = new_iou
                            if region_image == []:
                                pass
                            if offset[0]<0 or offset[1]<0:
                                pass
                            if region_image.shape[0]<=0 or region_image.shape[1]<=0:
                                break
                            history_vector = update_history_vector_for_state(history_vector, action)
                            region_image_vgg = np.expand_dims(cv2.resize(region_image, (224, 224)).astype(np.float32),
                                                              axis=0)
                            # 1*7*7*512
                            image_fea = sess.run(vgg.imgfea, feed_dict={vgg.image: region_image_vgg})
                            new_state = np.concatenate((np.reshape(image_fea, [-1]), history_vector), axis=0)

                            # Experience replay storage
                            if len(buffer) < buffer_experience_replay:
                                buffer.append((state, action, reward, new_state))
                            else:
                                if point < (buffer_experience_replay - 1):
                                    point += 1
                                else:
                                    point = 0

                                buffer[point] = (state, action, reward, new_state)
                                minibatch = random.sample(buffer, batch_size)
                                X_train = []
                                y_train = []
                                # we pick from the replay memory a sampled minibatch and generate the training samples
                                for memory in minibatch:
                                    old_state, action, reward, new_state = memory
                                    new_state_in = np.expand_dims(new_state,axis=0)
                                    A = sess.run(mainQN.predict,feed_dict={mainQN.state_input:new_state_in})
                                    Q = sess.run(targetQN.qval,feed_dict={targetQN.state_input:new_state_in})
                                    doubleQ = Q[0][A]
                                    if action != 10:
                                        targetQ = reward + gamma*doubleQ
                                    else:
                                        targetQ = reward
                                    y = np.zeros([1, 10])
                                    y = Q
                                    # if i%100 == 0:
                                    #     print y
                                    y = y.T

                                    y[action-1] = targetQ
                                    X_train.append(old_state)
                                    y_train.append(y)
                                X_train = np.array(X_train)
                                y_train = np.array(y_train)
                                X_train = X_train.astype("float32")
                                y_train = y_train.astype("float32")
                                X_train = X_train[:, :]
                                y_train = y_train[:, :, 0]
                                _,lr_rate,loss = sess.run([mainQN.train_op,mainQN.lr_rate,mainQN.cost],
                                                  feed_dict={mainQN.state_input:X_train,
                                                             mainQN.target_Q:y_train,
                                                             mainQN.global_step:i
                                                             })
                                print lr_rate
                                print('%s epoch for %s image with %s steps : cost : %s\n' % (i, j, step, loss))
                                t = get_weights()
                                a = sess.run(t[16:])


                                # update target DQN net
                                updateTarget(targetOps,sess)
                                aa = sess.run(t[16:])
                                state = new_state

                            if action == 10:
                                status = 0
                                masked = 1
                                # we mask object found with ground-truth so that agent learns faster
                                image_for_search = mask_image_with_mean_background(gt_mask, image_for_search)
                                # remain_objects[index] = 0

                            else:
                                masked = 0
                        remain_objects[index] = 0

            if epsilon > 0.1:
                epsilon -= 0.1

            if (i % 2 == 0):
                saver.save(sess, '/home/qs/models/Double_Duel/' + 'qnetmodel_aero.ckpt', global_step=i + 1)
            else:
                pass




