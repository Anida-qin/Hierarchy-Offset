import cv2
import numpy as np

BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    if data_set_name.startswith("aeroplane") | data_set_name.startswith("bird") | data_set_name.startswith("cow"):
        return [x.split(None, 1)[0] for x in image_names]
    else:
        return [x.strip('\n') for x in image_names]


def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    labels_names = f.readlines()
    labels_names = [x.split(None, 1)[1] for x in labels_names]
    labels_names = [x.strip('\n') for x in labels_names]
    return labels_names

def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        img = cv2.imread(string)
        img = img.astype(np.float32, copy=False)
        # img = cv2.resize(img, (224, 224)).astype(np.float32)
        img -= BGR_MEANS
        #print img.shape
        images.append(img)
    return images

def get_one_image(image_name, path_voc):
    string = path_voc + '/JPEGImages/' + image_name + '.jpg'
    img = cv2.imread(string)
    img = img.astype(np.float32, copy=False)
    # img = cv2.resize(img, (224, 224)).astype(np.float32)
    img -= BGR_MEANS
    img = np.array(img)
    return img

def generate_bbox_pixel_from_per_annotation(annotation, image_shape):
    length_annotation = annotation.shape[0]
    gt_masks = np.zeros([image_shape[0], image_shape[1], length_annotation])
    for i in range(0, length_annotation):
        gt_masks[int(annotation[i, 3]):int(annotation[i, 4]), int(annotation[i, 1]):int(annotation[i, 2]), i] = 1
    return gt_masks


# image_names = load_images_names_in_data_set('trainval',"/media/disk/ObDataset/VOC2007_trainval/")
# print image_names[0][1]
# print image_names[0]
# #labels_names = load_images_labels_in_data_set('trainval',"/media/disk/ObDataset/VOC2007_trainval/")
# #labels_names = load_images_labels_in_data_set('trainval',"/media/disk/ObDataset/VOC2007_trainval/")
# a=[]

def mask_image_with_mean_background(mask_object_found, image):
    new_image = image
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image