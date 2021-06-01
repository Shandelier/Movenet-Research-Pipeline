import glob
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import drawing_util as du


def count_files(path):
    path = path + "\*.jpg"
    return len(glob.glob(path))


def append_part(arr, path):
    df = pd.DataFrame(arr)
    df.to_csv(path, encoding="utf-8", index=False, mode="a", header=False)
    return 0


def create_file_name(
    path=r"C:\Users\kluse\Documents\python\posenet-python\output" + "\\",
):
    today = datetime.now()
    # return path + today.strftime("%d-%b-%Y-%H-%M") + ".csv"
    return path + today.strftime("%d-%b-%Y") + ".csv"


def create_log_file(path):
    names = ["postureType", "predScore"]
    for i, pn in enumerate(du.KEYPOINT_DICT):
        if i == 7:
            break
        names.append(pn + "X")
    for i, pn in enumerate(du.KEYPOINT_DICT):
        if i == 7:
            break
        names.append(pn + "Y")
    names = np.asarray([names])
    np.savetxt(path, names, delimiter=",", encoding="utf-8", fmt="%s")


KEYPOINT_LABELS = np.array([['filepath',
                             'pose_type',
                             'x_nose',
                             'x_left_eye',
                             'x_right_eye',
                             'x_left_ear',
                             'x_right_ear',
                             'x_left_shoulder',
                             'x_right_shoulder',
                             'x_left_elbow',
                             'x_right_elbow',
                             'x_left_wrist',
                             'x_right_wrist',
                             'x_left_hip',
                             'x_right_hip',
                             'x_left_knee',
                             'x_right_knee',
                             'x_left_ankle',
                             'x_right_ankle',
                             'y_nose',
                             'y_left_eye',
                             'y_right_eye',
                             'y_left_ear',
                             'y_right_ear',
                             'y_left_shoulder',
                             'y_right_shoulder',
                             'y_left_elbow',
                             'y_right_elbow',
                             'y_left_wrist',
                             'y_right_wrist',
                             'y_left_hip',
                             'y_right_hip',
                             'y_left_knee',
                             'y_right_knee',
                             'y_left_ankle',
                             'y_right_ankle',
                             'pred_nose',
                             'pred_left_eye',
                             'pred_right_eye',
                             'pred_left_ear',
                             'pred_right_ear',
                             'pred_left_shoulder',
                             'pred_right_shoulder',
                             'pred_left_elbow',
                             'pred_right_elbow',
                             'pred_left_wrist',
                             'pred_right_wrist',
                             'pred_left_hip',
                             'pred_right_hip',
                             'pred_left_knee',
                             'pred_right_knee',
                             'pred_left_ankle',
                             'pred_right_ankle', ]])
