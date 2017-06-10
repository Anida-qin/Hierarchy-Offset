import cv2
import numpy as np


def string_for_action(action):
    if action == 0:
        return "START"
    if action == 1:
        return 'up-left'
    elif action == 2:
        return 'up-right'
    elif action == 3:
        return 'down-left'
    elif action == 4:
        return 'down-right'
    elif action == 5:
        return 'center'
    elif action == 6:
        return 'TRIGGER'

def draw_sequence(step, action, reward, image_name, offset, region_image, path_to_save,image,draw_new,color):
    if draw_new:
        img_to_draw = image.copy()
    else:
        img_to_draw = image

    #region_shape = region_image.shape()
    h = region_image.shape[0]
    w = region_image.shape[1]
    # cols,rows
    top_left =(int(offset[1]),int(offset[0]))
    bottom_right =(int(offset[1]+w),int(offset[0]+h))
    img = cv2.rectangle(img=img_to_draw,pt1=top_left,pt2=bottom_right,color=color,thickness=3)
    #img = cv2.rectangle(img_to_draw, top_left, bottom_right,(0,0,255),3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    action_string = string_for_action(action)
    info_string =  'action: ' + action_string + ' ' + 'reward: ' + str(reward)

    cv2.putText(img_to_draw, info_string, (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    img_save_name = path_to_save + str(image_name) + '_'+ str(step)+'.png'
    cv2.imwrite(img_save_name,img)





#draw_sequence(1,1,1,'test_draw',[0,0],np.random.random((60,80)),'/home/qs/',np.zeros((224,224,3)))


