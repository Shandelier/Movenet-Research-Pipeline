import tensorflow_model_analysis as tfma
import apache_beam as beam

import tensorflow as tf
print('TF version: {}'.format(tf.__version__))
print('Beam version: {}'.format(beam.__version__))
print('TFMA version: {}'.format(tfma.__version__))


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
