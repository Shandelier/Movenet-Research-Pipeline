import tensorflow_model_analysis as tfma
import numpy as np
from matplotlib import pyplot as plt
import tensorboard as tb
import pandas as pd
from scipy import stats
import seaborn as sns
from packaging import version


def main():
    print("gmean")
    metrics = pd.read_csv(r"results\history_model_1_layer_128.csv")
    history = metrics.copy()
    tp = history.pop("val_tp")
    tn = history.pop("val_tn")
    fp = history.pop("val_fp")
    fn = history.pop("val_fn")
    precision = history.pop("val_precision")
    recall = history.pop("val_recall")

    fscore = 2 * (precision * recall) / (precision + recall)
    metric = tfma.metrics.F1Score(name="fscore")

    fscore2 = metric.result(tp, tn, fp, fn)
    print(fscore)
    print(fscore2)
    # history['fscore'] = fscore

    # print(history)


# def gmean(recall, precision):
#     metric = tfa.metrics.GeometricMean()
#     metric.update_state()
#     metric.result().numpy()
#     return 0


main()
