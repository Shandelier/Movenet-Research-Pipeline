import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import util as ut
import training_util as tut
import tensorflow as tf
import os
import json
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128


def train(csvs, output, results, final_results, epochs):
    if (csvs == None):
        csvs, _, _ = ut.get_csvs_paths(output)

    np.set_printoptions(precision=3, suppress=True)

    # result cube dimensions
    c_models = 3
    c_metrics = len(tut.skl_metrics)
    folds = 2
    repeats = 5
    rescube = np.zeros((folds*repeats, c_models, c_metrics))

    X, y, split = load_split(csvs, folds, repeats)
    for split_n, (train_index, test_index) in tqdm(enumerate(split.split(X, y)), desc="CrossVal", ascii=True, total=folds*repeats):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.2, random_state=420)

        train = tf.data.Dataset.from_tensor_slices(
            (X_train.values, y_train.values)).batch(BATCH_SIZE)
        validate = tf.data.Dataset.from_tensor_slices(
            (X_val.values, y_val.values)).batch(BATCH_SIZE)
        test = tf.data.Dataset.from_tensor_slices(
            (X_test.values, y_test.values)).batch(BATCH_SIZE)

        y_test = y_test.to_numpy()

        models, model_names = tut.get_models_and_names()

        for model_n, (model, model_name) in tqdm(enumerate(zip(models, model_names)), desc="Model", ascii=True, total=3, leave=False):
            history_logger, _ = loggers(
                results, model_name)
            tensorboard_logger = get_tensorboard_name(model_name)

            model.fit(train, epochs=epochs, callbacks=[
                history_logger, tensorboard_logger], validation_data=validate, verbose=0)

            pred = np.array(model.predict(test)).ravel()
            pred[:] = pred[:] >= 0.5

            model_scores = []
            for m in tut.skl_metrics:
                if m == 'gmean' or m == 'fscore':
                    model_scores.append(tut.skl_metrics[m](
                        pred, y_test, average='macro'))
                    continue
                model_scores.append(tut.skl_metrics[m](pred, y_test))
            try:
                rescube[split_n, model_n, :] = model_scores
            except:
                rescube[split_n, model_n, :] = np.nan
                print("WARNING: rescube subtable error")

    additional_metrics(results, final_results)
    np.save(os.path.join(results, "rescube"), rescube)

    with open(os.path.join(results, "legend.json"), "w") as outfile:
        json.dump(
            {
                "models": list(model_names),
                "metrics": list(tut.skl_metrics.keys()),
                "folds": folds,
                "repeats": repeats
            },
            outfile,
            indent="\t",
        )
    print(rescube)
    print(rescube.shape)
    return folds*repeats


def load_split(csvs, folds, repeats):
    ds, ds_labels = read_csvs(csvs)
    from sklearn.model_selection import RepeatedStratifiedKFold
    split = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=420)
    return ds, ds_labels, split


def loggers(results, model_name):
    history_logger = tf.keras.callbacks.CSVLogger(
        "{}/metrics_{}.csv".format(results, model_name), separator=",", append=True)

    validation_logger = tf.keras.callbacks.CSVLogger(
        "{}/validation_{}.csv".format(results, model_name), separator=",", append=True)
    validation_logger.on_test_begin = validation_logger.on_train_begin
    validation_logger.on_test_batch_end = validation_logger.on_epoch_end
    validation_logger.on_test_end = validation_logger.on_train_end

    return history_logger, validation_logger


def get_tensorboard_name(model_name):
    logdir = "logs/{}/".format(model_name) + \
        datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    return tensorboard_callback


def additional_metrics(results, final_results):
    csv_list, csv_names, _ = ut.get_csvs_paths(results)
    for i, (csv, name) in tqdm(enumerate(zip(csv_list, csv_names)), desc="File", ascii=True, total=len(csv_list)):
        metrics = pd.read_csv(csv)
        history = metrics.copy()
        precision = history.pop("precision")
        recall = history.pop("recall")
        val_precision = history.pop("val_precision")
        val_recall = history.pop("val_recall")

        fscore = 2 * (precision * recall) / (precision + recall)
        val_fscore = 2 * (val_precision * val_recall) / \
            (val_precision + val_recall)
        metrics['fscore'] = fscore
        metrics['val_fscore'] = val_fscore
        metrics.to_csv(os.path.join(final_results, name+".csv"))


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
