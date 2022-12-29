import cv2
import time
import drawing_util as du
import tensorflow as tf
import numpy as np
import time
import cropping as cr
import movenet as mvn

movenet, input_size = mvn.choose_model(model_name="li")



cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret_val, img = cam.read()
convert_img = tf.convert_to_tensor(img)
crop_region = cr.init_crop_region(convert_img.shape[0], convert_img.shape[1])

for i in range(100):
    ret_val, img = cam.read()
    img = cv2.resize(img, (256, 192))
    convert_img = tf.convert_to_tensor(img)

    keypoints_with_scores = cr.run_inference(
        movenet, convert_img[:,:,:], crop_region,
        crop_size=[input_size, input_size])

    flatten = np.hstack([keypoints_with_scores.squeeze().T.flatten()]).reshape([1, 51])



    crop_region = cr.determine_crop_region(
        keypoints_with_scores, convert_img.shape[0], convert_img.shape[1])

    cool_img = du.draw_prediction_on_image(
        img, keypoints_with_scores, crop_region, close_figure=False,
        output_image_height=convert_img.shape[0])

    cv2.imshow('my webcam', cool_img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
