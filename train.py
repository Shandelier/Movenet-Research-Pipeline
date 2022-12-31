from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import util as ut
import training_util as tut
import tensorflow as tf
import os
import json
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import wandb


BATCH_SIZE = 128
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# wandb.init(project="Pose", sync_tensorboard=True)


def train(output, result_path, epochs=10):
    csvs, csvnames, _ = ut.get_csvs_paths(output)

    model_n = 3
    ds_n = len(csvnames)
    experiment_n = model_n * ds_n

    # result cube dimensions
    c_models = 3
    c_metrics = len(tut.skl_metrics)
    folds = 2
    repeats = 1
    rescube = np.zeros((folds*repeats, c_models, c_metrics))

    for c, data_name in zip(csvs, csvnames):
        ds = pd.read_csv(c)
        sample_filepath = ds.pop('filepath')
        y = ds.pop('pose_type')
        X = ds.copy()

        split = RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=repeats, random_state=420)

        first_layer = X.shape[1]

        models, model_names = tut.get_models_and_names(
            first_layer=first_layer)

        for model, model_name in tqdm(zip(models, model_names), desc="Model", ascii=True, total=3, leave=False):
            experiment_name = model_name+"_on_"+data_name

            for split_n, (train_index, test_index) in tqdm(enumerate(split.split(X, y)), desc="CrossVal", ascii=True, total=folds*repeats):
                # Split the data into train and test sets
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Convert the data to TensorFlow datasets
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_train, y_train))
                test_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_test, y_test))

                # Batch the datasets
                train_dataset = train_dataset.batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE)

                model.fit(train_dataset, epochs=epochs, callbacks=[
                    get_tensorboard_name(experiment_name),
                    # get_wandb(epochs),
                    get_history_logger(
                        result_path, f"{experiment_name}")
                ],
                    validation_data=test_dataset, verbose=0)
            model.save('models/classifier/{}.h5'.format(experiment_name))
    # wandb.finish()


def get_tensorboard_name(model_name):
    logdir = "logs/{}/".format(model_name) + \
        datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    return tensorboard_callback


def get_wandb(epochs):
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": BATCH_SIZE
    }
    return wandb.tensorflow.WandbHook(steps_per_log=epochs/10)


def get_history_logger(results, model_name):
    return tf.keras.callbacks.CSVLogger(
        "{}/metrics_{}.csv".format(results, model_name), separator=",", append=True)


def read_csvs(csvs):
    init = list.pop(csvs)
    ds = pd.read_csv(init)
    for _, csv in tqdm(enumerate(csvs), desc="Merging CSVs", ascii=True, total=len(csvs)):
        read = pd.read_csv(csv)
        ds = pd.concat([ds, read], axis=0)

    print("Samples and features: {}".format(ds.shape))
    class_sizes = ds['pose_type'].value_counts()
    print("Class 0 = {}, Class 1 = {}".format(class_sizes[0], class_sizes[1]))
    smaller_class = min(class_sizes)
    bigger_class = max(class_sizes)
    print("Class inequality index {}. Class inequal by {} samples".format(
        bigger_class/smaller_class, bigger_class-smaller_class))
    print("Applying undresampling to balance the dataset")
    ds = ds.groupby('pose_type').apply(
        lambda x: x.sample(smaller_class, random_state=420))

    from sklearn.utils import shuffle
    ds = shuffle(ds, random_state=420)
    ds = ds.reset_index(drop=True)

    sample_filepath = ds.pop('filepath')
    ds_labels = ds.pop('pose_type')

    ds = ds.astype(dtype=np.float32)
    ds_labels = ds_labels.astype(dtype=np.float32)
    return ds, ds_labels


# train(None, r"./ds", r'./results')
