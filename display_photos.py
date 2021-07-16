from sklearn.utils import shuffle
import util as ut
import os
import pandas as pd
import cv2
from tqdm import tqdm

source_p = os.path.join('output')
pictures_p = os.path.join(
    'C:\\Users\\kluse\\Documents\\python\\SSR-Dataset\\pictures_700K')

resize_scale = 4

part_name_x = [
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
    'x_right_ankle']

part_name_arr = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle']

excessive_pred = [
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
    'pred_right_ankle']

excessive = [
    'x_left_hip',
    'x_right_hip',
    'x_left_knee',
    'x_right_knee',
    'x_left_ankle',
    'x_right_ankle',
    'y_left_hip',
    'y_right_hip',
    'y_left_knee',
    'y_right_knee',
    'y_left_ankle',
    'y_right_ankle']


def plot_image(file_path, y, x):
    index = 0
    while(True):
        img_path = os.path.join(
            pictures_p, file_path[index]).replace("/", "\\")
        img = cv2.imread(img_path, 1)

        alternate_size = (img.shape[1]*resize_scale, img.shape[0]*resize_scale)
        img = cv2.resize(img, (alternate_size))

        x_coor = (x[index, :]*img.shape[1]).astype(int)
        y_coor = (y[index, :]*img.shape[0]).astype(int)

        cv2.putText(img, file_path[index],
                    (10, 18), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (66, 245, 66), 1, cv2.LINE_AA)

        for i, (w, h) in enumerate(zip(x_coor, y_coor)):
            cv2.circle(img, (w, h),
                       2, (255, 161, 239), -1)

            cv2.putText(img, part_name_arr[i],
                        (w+10, h+10), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, (255, 161, 239), 1, cv2.LINE_AA)

        cv2.imshow(f'Sit Stand Right Preview', img)
        key = cv2.waitKey(0)

        if key == ord('w'):
            print("\tCopying this one")
        elif key == ord('a'):
            if index > 0:
                index -= 1
        elif key == ord('d'):
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
for i, csv in tqdm(enumerate(file_paths), desc="Merging CSVs", ascii=True, total=len(file_paths)):
    read = pd.read_csv(csv)
    ds = pd.concat([ds, read], axis=0)
ds = shuffle(ds)
ds = ds.reset_index(drop=True)

for p in excessive_pred:
    ds.pop(p)
# for e in excessive:
#     ds.pop(e)
labels = ds.pop('pose_type')
file_path = ds.pop('filepath')

x = ds.pop(part_name_x.pop(0))
for col in part_name_x:
    colu = ds.pop(col)
    x = pd.concat([x, colu], axis=1)

# print(file_path)
plot_image(file_path, x.to_numpy(), ds.to_numpy())
print("over")
