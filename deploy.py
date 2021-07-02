import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import util as ut
import training_util as tut


def deploy(csvs=None, output=os.path.join("output"), results=os.path.join("saved_models"), epochs=100):
    if (csvs == None):
        csvs, _, _ = ut.get_csvs_paths(output)

    np.set_printoptions(precision=3, suppress=True)

    if not os.path.exists(os.path.join("saved_models")):
        os.makedirs(os.path.join("saved_models"))

    # result cube dimensions
    c_models = 3
    c_metrics = len(tut.skl_metrics)
    folds = 2
    repeats = 5
    rescube = np.zeros((folds*repeats, c_models, c_metrics))

    X, y, split = load_split(csvs, folds, repeats)
    ds = tf.data.Dataset.from_tensor_slices(
        (X.values, y.values)).batch(128)

    models, model_names = tut.get_models_and_names()

    for model_n, (model, model_name) in tqdm(enumerate(zip(models, model_names)), desc="Model", ascii=True, total=len(models), leave=False):
        history_logger = loggers(results, model_name)
        model.fit(ds, epochs=epochs, callbacks=[
            history_logger], verbose=0)
        model.save(os.path.join("saved_models", model_name+".h5"))
        tfjs.converters.save_keras_model(
            model, os.path.join("saved_model_js", model_name))


def load_split(csvs, folds, repeats):
    ds, ds_labels = read_csvs(csvs)
    from sklearn.model_selection import RepeatedStratifiedKFold
    split = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=420)
    return ds, ds_labels, split


def loggers(results, model_name):
    history_logger = tf.keras.callbacks.CSVLogger(
        "{}/metrics_{}.csv".format(results, model_name), separator=",", append=True)
    return history_logger


def read_csvs(csvs):
    init = list.pop(csvs)
    ds = pd.read_csv(init)
    for i, csv in enumerate(csvs):
        read = pd.read_csv(csv)
        ds = pd.concat([ds, read], axis=0)

    ds = ds.reset_index(drop=True)
    from sklearn.utils import shuffle
    ds = shuffle(ds, random_state=420)

    count = ds['pose_type'].value_counts()

    sample_filepath = ds.pop('filepath')
    ds_labels = ds.pop('pose_type')

    for p in tut.excessive_pred:
        ds.pop(p)
    for e in tut.excessive:
        ds.pop(e)

    print("Samples and features: {}".format(ds.shape))
    print("Class 0 = {}, Class 1 = {}".format(count[0], count[1]))

    ds = ds.astype(dtype=np.float32)
    ds_labels = ds_labels.astype(dtype=np.float32)
    return ds, ds_labels


def default_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(51),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam())
    model.build([1, 1, 17, 3])
    model.save(os.path.join("saved_models", "default.h5"))
    tfjs.converters.save_keras_model(
        model, os.path.join("saved_models", "default-js"))


deploy()
