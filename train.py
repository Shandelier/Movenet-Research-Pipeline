import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import conv2d, dropout
from tensorflow.python.keras.layers.core import Dropout

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def train(csvs, output):
    np.set_printoptions(precision=3, suppress=True)
    ds = load_csvs(csvs)

    sample_filepath = ds.pop('filepath')
    ds_labels = ds.pop('pose_type')

    pred = [
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

    for p in pred:
        ds.pop(p)
    for e in excessive:
        ds.pop(e)

    ds_features = ds.copy()

    print(ds_features)
    print(ds_labels)

    norm_ssr_model = tf.keras.Sequential([
        # normalize,
        layers.Dense(22),
        layers.Dense(128),
        layers.Dense(64),
        layers.Dense(32),
        # layers.Dropout(.2),
        layers.Dense(16),
        layers.Dense(8),
        # layers.Dropout(.2),
        layers.Dense(1)
    ])

    norm_ssr_model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics='acc')

    history = norm_ssr_model.fit(ds_features, ds_labels, epochs=250)

    acc = history.history['acc']
    loss = history.history['loss']

    print(acc)

    np.savetxt(output+'/acc.csv', acc, delimiter=",")
    np.savetxt(output+'/loss.csv', loss, delimiter=",")


def load_csvs(csvs):
    init = list.pop(csvs)
    ds = pd.read_csv(init)
    for i, csv in enumerate(csvs):
        read = pd.read_csv(csv)
        ds = pd.concat([ds, read])
    return ds
