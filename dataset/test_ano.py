import cv2, numpy as np

from img_helper import *


if __name__ == "__main__":
    ######## PATHS definition ########

    # path of pascal voc test
    path_voc_test = "/media/disk/ObDataset/VOC2007_test/"
    # model name of the weights
    #model_name = "model_image_zooms"
    # path of folder where the weights are
    #weights_path = "../models_image_zooms/"
    # path of where to store visualizations of search sequences
    #path_testing_folder = '../testing/'
    # path of VGG16 weights
    #path_vgg = "../vgg16_weights.h5"

    ######## MODELS ########

    #model_vgg = obtain_compiled_vgg_16(path_vgg)
    #model = get_q_network(weights_path + model_name)

    ######## LOAD IMAGE NAMES ########

    image_names = np.array([load_images_names_in_data_set('aeroplane_test', path_voc_test)])
    labels = load_images_labels_in_data_set('aeroplane_test', path_voc_test)

    ######## LOAD IMAGES ########

    images = get_all_images(image_names, path_voc_test)

    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3) / 4
    scale_mask = float(1) / (scale_shopvubregion * 4)
    # Number of steps that the agent does at each image
    number_of_steps = 10
    # Only search first object
    only_first_object = 1