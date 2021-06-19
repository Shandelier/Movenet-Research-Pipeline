import tensorflow_model_analysis as tfma
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import util as ut
import os
import tensorboard as tb
import pandas as pd
from scipy import stats
import seaborn as sns
from packaging import version


def main():
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
        fscore = 2 * (precision * recall) / (precision + recall)
        # TODO: gmean i BAC
        metrics['fscore'] = fscore
        metrics.to_csv(os.path.join("final_results", name+".csv"))

    # history['fscore'] = fscore

    # print(history)


# def gmean(recall, precision):
#     metric = tfa.metrics.GeometricMean()
#     metric.update_state()
#     metric.result().numpy()
#     return 0
main()
