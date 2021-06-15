import pandas as pd
import numpy as np
from tqdm import tqdm
import util as ut
import training_util as tut


def train(csvs, output, results):
    if (csvs == None):
        csvs, _, _ = ut.get_csvs_paths(output)

    np.set_printoptions(precision=3, suppress=True)
    ds = load_csvs(csvs)

    sample_filepath = ds.pop('filepath')
    ds_labels = ds.pop('pose_type')

    for p in tut.excessive_pred:
        ds.pop(p)
    for e in tut.excessive:
        ds.pop(e)

    ds_features = ds.copy()

    print(ds_features)
    print(ds_labels)

    models, model_names = tut.get_models_and_names()

    for model, model_name in tqdm(zip(models, model_names),
                                  desc="MODEL", ascii=True, total=len(models)):
        ds_features = ds.copy()
        history = model.fit(ds_features, ds_labels, epochs=10)

        accuracy = history.history['accuracy']
        precision = history.history['precision']
        recall = history.history['recall']
        # fscore = history.history['fscore']
        # gmean = history.history['gmean']
        loss = history.history['loss']
        tp = history.history['tp']
        fp = history.history['fp']
        tn = history.history['tn']
        fn = history.history['fn']

        np.savetxt("{}/accuracy_{}.csv".format(results,
                   model_name), accuracy, delimiter=",")
        np.savetxt("{}/precision_{}.csv".format(results,
                   model_name), precision, delimiter=",")
        np.savetxt("{}/recall_{}.csv".format(results,
                   model_name), recall, delimiter=",")
        # np.savetxt("{}/gmean_{}.csv".format(results,
        #            model_name), gmean, delimiter=",")
        # np.savetxt("{}/fscore_{}.csv".format(results,
        #            model_name), fscore, delimiter=",")
        np.savetxt("{}/loss_{}.csv".format(results,
                   model_name), loss, delimiter=",")
        np.savetxt("{}/TruePositives_{}.csv".format(results,
                   model_name), tp, delimiter=",")
        np.savetxt("{}/FalsePositives_{}.csv".format(results,
                   model_name), fp, delimiter=",")
        np.savetxt("{}/TrueNegatives_{}.csv".format(results,
                   model_name), tn, delimiter=",")
        np.savetxt("{}/FalseNegatives_{}.csv".format(results,
                   model_name), fn, delimiter=",")


def load_csvs(csvs):
    init = list.pop(csvs)
    ds = pd.read_csv(init)
    for i, csv in enumerate(csvs):
        read = pd.read_csv(csv)
        ds = pd.concat([ds, read], axis=0)
    return ds
