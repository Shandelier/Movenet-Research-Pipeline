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


import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./video')
args = parser.parse_args()

vids = args.video
print(vids)

file_paths = [f.path for f in os.scandir(
    input) if f.is_file() and f.path.endswith('.csv')]
print(file_paths)
