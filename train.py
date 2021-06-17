import pandas as pd
import numpy as np
from tqdm import tqdm
import util as ut
import training_util as tut
import tensorflow as tf
import os


def train(csvs, output, results, final_results, epochs):
    if (csvs == None):
        csvs, _, _ = ut.get_csvs_paths(output)

    np.set_printoptions(precision=3, suppress=True)

    X_train, y_train, X_test, y_test, X_val, y_val = load_csvs(csvs)

    models, model_names = tut.get_models_and_names()

    for model, model_name in tqdm(zip(models, model_names),
                                  desc="MODEL", ascii=True, total=len(models)):
        print("MODEL: ", model_name)

        history_logger, validation_logger = loggers(results, model_name)
        model.fit(X_train, y_train, epochs=epochs,
                  validation_data=(X_test, y_test), callbacks=[history_logger], verbose=1)
        model.evaluate(
            X_val, y_val, callbacks=[validation_logger], return_dict=True, verbose=0)

    # Save additional metrics
    additional_metrics(results, final_results)


def load_csvs(csvs):
    init = list.pop(csvs)
    ds = pd.read_csv(init)
    for i, csv in enumerate(csvs):
        read = pd.read_csv(csv)
        ds = pd.concat([ds, read], axis=0)

    sample_filepath = ds.pop('filepath')
    ds_labels = ds.pop('pose_type')

    for p in tut.excessive_pred:
        ds.pop(p)
    for e in tut.excessive:
        ds.pop(e)

    ds = ds.astype(dtype=np.float32)
    ds_labels = ds_labels.astype(dtype=np.float32)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        ds, ds_labels, test_size=0.4, random_state=420)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=420)
    return X_train, y_train, X_test, y_test, X_val, y_val


def loggers(results, model_name):
    history_logger = tf.keras.callbacks.CSVLogger(
        "{}/history_{}.csv".format(results, model_name), separator=",", append=True)

    validation_logger = tf.keras.callbacks.CSVLogger(
        "{}/validation_{}.csv".format(results, model_name), separator=",", append=True)
    validation_logger.on_test_begin = validation_logger.on_train_begin
    validation_logger.on_test_batch_end = validation_logger.on_epoch_end
    validation_logger.on_test_end = validation_logger.on_train_end

    return history_logger, validation_logger


def additional_metrics(results, final_results):
    csv_list, csv_names, _ = ut.get_csvs_paths(r"./results")
    for i, (csv, name) in tqdm(enumerate(zip(csv_list, csv_names)), desc="File", ascii=True, total=len(csv_list)):
        metrics = pd.read_csv(csv)
        history = metrics.copy()
        tp = history.pop("tp")
        tn = history.pop("tn")
        fp = history.pop("fp")
        fn = history.pop("fn")
        precision = history.pop("precision")
        recall = history.pop("recall")
        # TODO: gmean and BAC
        fscore = 2 * (precision * recall) / (precision + recall)
        metrics['fscore'] = fscore
        metrics.to_csv(os.path.join("full_results", name+".csv"))
