import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import numpy as np
import argparse
import os


def display_graph(results, model_names, metric_name):
    plt.figure(figsize=(8, 6))
    epochs = np.linspace(0, len(results[0]), len(results[0]))

    for i, (result, model_name) in enumerate(zip(results, model_names)):
        plt.plot(epochs, result, label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.ylim([0, 1])
        plt.legend()


def main():
    metrics = ['accuracy', 'precision', 'recall', 'fscore', 'loss']
    csv_list, csv_names, _ = ut.get_csvs_paths(r"./final_results")
    history = []
    for csv in csv_list:
        history.append(pd.read_csv(csv))

    for m in metrics:
        metric_result_list = []
        model_result_names = []
        for i, (h, name) in enumerate(zip(history, csv_names)):
            metric_result_list.append(h.pop(m))
            model_result_names.append(name)
        display_graph(metric_result_list, model_result_names, m)


main()
