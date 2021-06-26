# import tensorflow as tf
# import tensorflow_addons as tfa
# # import tensorflow_model_analysis as tfma
# import numpy as np
# import pandas as pd

# print('TF version: {}'.format(tf.__version__))
# # print('TFMA version: {}'.format(tfma.__version__))


# metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
# # # y_true = np.array([[1, 1, 1],
# # #                    [1, 0, 0],
# #                    [1, 1, 0]], np.int32)
# # y_pred = np.array([[0.2, 0.6, 0.7],
# #                    [0.2, 0.6, 0.6],
# #                    [0.6, 0.8, 0.0]], np.float32)
# # metric.update_state(y_true, y_pred)
# # result = metric.result()
# # print(result.numpy())

# tp = pd.read_csv(
#     r'./results/TruePositives_model_1_layer_128.csv', header=None).to_numpy().flatten()
# tn = pd.read_csv(
#     r'./results/TrueNegatives_model_1_layer_128.csv', header=None).to_numpy().flatten()
# fp = pd.read_csv(
#     r'./results/FalsePositives_model_1_layer_128.csv', header=None).to_numpy().flatten()
# fn = pd.read_csv(
#     r'./results/FalseNegatives_model_1_layer_128.csv', header=None).to_numpy().flatten()
# some_recall = pd.read_csv(
#     r'./results/recall_model_1_layer_128.csv', header=None).to_numpy().flatten()

# # print(tp)
# # print(tn)
# # print(fp)
# # print(fn)

# # tp / (tp + fn)
# recall = tp / (tp + fn)
# print(recall)
# print(some_recall)

# import time
# import argparse
# import os
# from tqdm import tqdm

# # Import matplotlib libraries
# from matplotlib import pyplot as plt

# import util
# import drawing_util as du
# import cropping as cr

# filenames = [
#     f.path for f in os.scandir('./input') if f.is_file() and f.path.endswith(('.png', '.jpg'))]

# for fname in tqdm(filenames, desc="FILE", ascii=True, total=5):
#     # Load the input image.
#     print(fname)

# import pandas as pd
# import numpy as np

# tp = pd.read_csv(r'results\TruePositivesmodel_1_layer_128.csv', header=None)
# fn = pd.read_csv(r'results\FalsePositivesmodel_1_layer_128.csv', header=None)
# recall = pd.read_csv(r'results\recall_model_1_layer_128.csv', header=None)

# print(tp)
# print(fn)
# print(recall)

# from packaging import version

# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# from scipy import stats
# import tensorboard as tb

# major_ver, minor_ver, _ = version.parse(tb.__version__).release
# assert major_ver >= 2 and minor_ver >= 3, \
#     "This notebook requires TensorBoard 2.3 or later."
# print("TensorBoard version: ", tb.__version__)

# experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
# print(df)
# print(df["run"].unique())
# print(df["tag"].unique())

# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# sns.lineplot(data=dfw_validation, x="step", y="epoch_accuracy",
#              hue=optimizer_validation).set_title("accuracy")
# plt.subplot(1, 2, 2)
# sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
#              hue=optimizer_validation).set_title("loss")


from sklearn.utils import shuffle
import util as ut
import os
import pandas as pd
import training_util as tut
import numpy as np

file_paths, file_names, pose_type = ut.get_csvs_paths(
    os.path.join("5-people-csvs"))


init = list.pop(file_paths)
ds = pd.read_csv(init)
for i, csv in enumerate(file_paths):
    read = pd.read_csv(csv)
    ds = pd.concat([ds, read], axis=0)


ds.pop('filepath')
for p in tut.excessive_pred:
    ds.pop(p)
for e in tut.excessive:
    ds.pop(e)

ds = shuffle(ds, random_state=420)
ds.reset_index(drop=True)
label = ds.pop("pose_type")
lable = pd.Series(np.where(label.values == 1, "bad", "good"),
                  label.index)

ds = pd.concat([ds, label], axis=1)

small_csv = ds.sample(n=2000)

small_csv.to_csv(os.path.join(
    "results", "5_people_small.csv"), sep='\t', index=False, header=True)
print(small_csv.head())
# ds.to_csv(os.path.join(
#     "results", "5_people.csv"), sep='\t', index=False, header=False)
