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


csv_column_names = np.array(
    [
        [
            "predScore",
            "noseX",
            "leftEyeX",
            "rightEyeX",
            "leftEarX",
            "rightEarX",
            "leftShoulderX",
            "rightShoulderX",
            "noseY",
            "leftEyeY",
            "rightEyeY",
            "leftEarY",
            "rightEarY",
            "leftShoulderY",
            "rightShoulderY",
            "postureType",
        ]
    ]
)
