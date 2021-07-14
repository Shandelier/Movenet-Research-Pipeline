from sklearn.utils import shuffle
import util as ut
import os
import pandas as pd
import training_util as tut
import numpy as np
import cv2

source_p = os.path.join('jojo-csv')

XX = [
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
    'x_right_ankle', ]


def plot_image(file_path, x, y):
    index = 0
    while(True):
        img = cv2.imread(file_path[index], 1)

        x_coor = (x[index, :]*img.shape[1]).astype(int)
        y_coor = (y[index, :]*img.shape[0]).astype(int)

        for i, (w, h) in enumerate(zip(x_coor, y_coor)):
            cv2.circle(img, (w, h),
                       5, (255, 161, 239), -1)

        cv2.imshow(f'current image', img)
        key = cv2.waitKey(0)

        if key == ord('c'):
            print("\tCopying this one")
        elif key == ord('x'):
            if index > 0:
                index -= 1
        elif key == ord('v'):
            index += 1
        elif key == ord('q'):
            break

        cv2.destroyAllWindows()


def movenet_overlay(file_path):
    return 0


# Display frames of certain value in column
file_paths, file_names, pose_type = ut.get_csvs_paths(source_p)

init = list.pop(file_paths)
ds = pd.read_csv(init)
for i, csv in enumerate(file_paths):
    read = pd.read_csv(csv)
    ds = pd.concat([ds, read], axis=0)

ds = ds.reset_index(drop=True)
for p in tut.excessive_pred:
    ds.pop(p)
# for e in tut.excessive:
#     ds.pop(e)
labels = ds.pop('pose_type')
file_path = ds.pop('filepath')

x = ds.pop('x_nose')
for col in XX:
    colu = ds.pop(col)
    x = pd.concat([x, colu], axis=1)

# print(file_path)
plot_image(file_path, x.to_numpy(), ds.to_numpy())
print("over")
