import cv2
import time
import drawing_util as du
import tensorflow as tf
import numpy as np
import time
import cropping as cr
import movenet as mvn
import pandas as pd
import numpy as np

x = ["x_nose", "x_left_eye", "x_right_eye", "x_left_ear", "x_right_ear", "x_left_shoulder", "x_right_shoulder", "x_left_elbow", "x_right_elbow",
     "x_left_wrist", "x_right_wrist", "x_left_hip", "x_right_hip", "x_left_knee", "x_right_knee", "x_left_ankle", "x_right_ankle", ]
y = ["y_nose", "y_left_eye", "y_right_eye", "y_left_ear", "y_right_ear", "y_left_shoulder", "y_right_shoulder", "y_left_elbow", "y_right_elbow",
     "y_left_wrist", "y_right_wrist", "y_left_hip", "y_right_hip", "y_left_knee", "y_right_knee", "y_left_ankle", "y_right_ankle", ]
# distance
r = ["r_nose", "r_left_eye", "r_right_eye", "r_left_ear", "r_right_ear", "r_left_shoulder", "r_right_shoulder", "r_left_elbow", "r_right_elbow",
     "r_left_wrist", "r_right_wrist", "r_left_hip", "r_right_hip", "r_left_knee", "r_right_knee", "r_left_ankle", "r_right_ankle", ]
rnonose = ["r_left_eye", "r_right_eye", "r_left_ear", "r_right_ear", "r_left_shoulder", "r_right_shoulder", "r_left_elbow", "r_right_elbow",
           "r_left_wrist", "r_right_wrist", "r_left_hip", "r_right_hip", "r_left_knee", "r_right_knee", "r_left_ankle", "r_right_ankle", ]


def convert_to_distance(flatten):
    df = pd.DataFrame(flatten, columns=['x_nose', 'x_left_eye', 'x_right_eye',
                                        'x_left_ear', 'x_right_ear', 'x_left_shoulder', 'x_right_shoulder',
                                        'x_left_elbow', 'x_right_elbow', 'x_left_wrist', 'x_right_wrist',
                                        'x_left_hip', 'x_right_hip', 'x_left_knee', 'x_right_knee',
                                        'x_left_ankle', 'x_right_ankle', 'y_nose', 'y_left_eye', 'y_right_eye',
                                        'y_left_ear', 'y_right_ear', 'y_left_shoulder', 'y_right_shoulder',
                                        'y_left_elbow', 'y_right_elbow', 'y_left_wrist', 'y_right_wrist',
                                        'y_left_hip', 'y_right_hip', 'y_left_knee', 'y_right_knee',
                                        'y_left_ankle', 'y_right_ankle', 'pred_nose', 'pred_left_eye',
                                        'pred_right_eye', 'pred_left_ear', 'pred_right_ear',
                                        'pred_left_shoulder', 'pred_right_shoulder', 'pred_left_elbow',
                                        'pred_right_elbow', 'pred_left_wrist', 'pred_right_wrist',
                                        'pred_left_hip', 'pred_right_hip', 'pred_left_knee', 'pred_right_knee',
                                        'pred_left_ankle', 'pred_right_ankle'])

    xs = df.filter(like='x_')
    ys = df.filter(like='y_')
    # factor = df.filter(like='pred')
    xnose = xs['x_nose']
    ynose = ys['y_nose']
    xs = xs.drop(columns=['x_nose'])
    ys = ys.drop(columns=['y_nose'])

    df[rnonose] = (((np.array([xnose.values]*16) - xs.values.T)**2 +
                    (np.array([ynose.values]*16) - ys.values.T)**2)**.5).T

    return df[["x_nose", "y_nose", "pred_nose", "pred_left_eye", "pred_right_eye", "pred_left_ear", "pred_right_ear", "pred_left_shoulder", "pred_right_shoulder", "pred_left_elbow", "pred_right_elbow", "pred_left_wrist", "pred_right_wrist", "pred_left_hip", "pred_right_hip", "pred_left_knee", "pred_right_knee", "pred_left_ankle", "pred_right_ankle", "r_left_eye", "r_right_eye", "r_left_ear", "r_right_ear", "r_left_shoulder", "r_right_shoulder", "r_left_elbow", "r_right_elbow", "r_left_wrist", "r_right_wrist", "r_left_hip", "r_right_hip", "r_left_knee", "r_right_knee", "r_left_ankle", "r_right_ankle"]]


movenet, input_size = mvn.choose_model(model_name="li")
classifier = tf.keras.models.load_model('models/classifier/deep-distance.h5')

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret_val, img = cam.read()
convert_img = tf.convert_to_tensor(img)
crop_region = cr.init_crop_region(convert_img.shape[0], convert_img.shape[1])

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50, 50)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 1
lineType = 2

isDrawingSkeleton = False
isPredOnDistance = True

while True:
    ret_val, img = cam.read()
    cool_img = img = cv2.resize(img, (256, 192))
    convert_img = tf.convert_to_tensor(img)

    keypoints_with_scores = cr.run_inference(
        movenet, convert_img, crop_region,
        crop_size=[input_size, input_size])

    flatten = np.hstack(
        [keypoints_with_scores.squeeze().T.flatten()]).reshape([1, 51])

    if isPredOnDistance:
        distance = convert_to_distance(flatten)
        pred = classifier.predict(distance)
    else:
        pred = classifier.predict(flatten)

    crop_region = cr.determine_crop_region(
        keypoints_with_scores, convert_img.shape[0], convert_img.shape[1])

    if (isDrawingSkeleton):
        cool_img = du.draw_prediction_on_image(
            img, keypoints_with_scores, crop_region, close_figure=False,
            output_image_height=convert_img.shape[0])

    if (pred > 0.9):
        cv2.putText(cool_img, 'CAVEMAN!',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

    cv2.imshow('my webcam', cool_img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    if cv2.waitKey(1) == 49:
        isDrawingSkeleton = ~isDrawingSkeleton  # toggl drawing

cv2.destroyAllWindows()
