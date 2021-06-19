import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import numpy as np
import argparse
import os


def display_graph(results, model_names, metric_name, results_graphs):
    plt.figure(figsize=(8, 6))

    max_epoch = results[0].size
    for r in results:
        if max_epoch < r.size:
            max_epoch = r.size

    epochs = np.linspace(0, max_epoch, max_epoch)

    for i, (result, model_name) in enumerate(zip(results, model_names)):
        # print(result)
        # diff = len(epochs) - result.size
        # if diff != 0:
        #     zero = np.zeros(diff)
        #     array = result.values.tolist()
        #     array =
        #     result = result.append(zero)
        plt.plot(np.linspace(0, result.size, result.size),
                 result, label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.ylim([0, 1])
        plt.legend()
    plt.savefig(os.path.join(results_graphs,
                             '{}.png'.format(metric_name)))


def disp(results_final=r"./results_final", results_graphs=r"./results_graphs"):
    metrics = ['accuracy', 'precision', 'recall', 'fscore', 'loss']
    csv_list, csv_names, _ = ut.get_csvs_paths(results_final)
    history = []
    for csv in csv_list:
        history.append(pd.read_csv(csv))

    for m in metrics:
        metric_result_list = []
        model_result_names = []
        for i, (h, name) in enumerate(zip(history, csv_names)):
            metric_result_list.append(h.pop(m))
            model_result_names.append(name)
        display_graph(metric_result_list,
                      model_result_names, m, results_graphs)
