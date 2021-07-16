import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os
from tqdm import tqdm
import util as ut
import training_util as tut
import train as tr


def deploy(csvs=None, output=os.path.join("output"), results=os.path.join("saved_models"), epochs=20):
    if (csvs == None):
        csvs, _, _ = ut.get_csvs_paths(output)

    np.set_printoptions(precision=3, suppress=True)

    if not os.path.exists(results):
        os.makedirs(results)
    if not os.path.exists(results+"_js"):
        os.makedirs(results+"_js")

    X, y = tr.read_csvs(csvs)
    ds = tf.data.Dataset.from_tensor_slices(
        (X.values, y.values)).batch(128)

    models, model_names = tut.get_models_and_names()

    for _, (model, model_name) in tqdm(enumerate(zip(models, model_names)), desc="Model", ascii=True, total=len(models), leave=False):
        history_logger = loggers(results, model_name)
        model.fit(ds, epochs=epochs, callbacks=[
            history_logger], verbose=0)
        model.save(os.path.join(results, model_name+".h5"))
        tfjs.converters.save_keras_model(
            model, os.path.join(results+"_js", model_name))

    print("h5 models ready")


def loggers(results, model_name):
    history_logger = tf.keras.callbacks.CSVLogger(
        "{}/metrics_{}.csv".format(results, model_name), separator=",", append=True)
    return history_logger


deploy()
