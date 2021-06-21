import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import numpy as np
import argparse
import os


def display_graph(results, val_results, model_names, metric_name, results_graphs):
    plt.figure(figsize=(8, 6))

    max_epoch = results[0].size
    for r in results:
        if max_epoch < r.size:
            max_epoch = r.size

    epochs = np.linspace(0, max_epoch, max_epoch)

    for i, (result, val_result, model_name) in enumerate(zip(results, val_results, model_names)):
        # print(result)
        # diff = len(epochs) - result.size
        # if diff != 0:
        #     zero = np.zeros(diff)
        #     array = result.values.tolist()
        #     array =
        #     result = result.append(zero)

        plt.plot(np.linspace(0, result.size, result.size),
                 result, label=model_name)
        plt.plot(np.linspace(0, result.size, result.size),
                 val_result, label="val_"+model_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.ylim([0, 1])
        plt.legend()
    plt.savefig(os.path.join(results_graphs,
                             '{}.png'.format(metric_name)))


def disp(results_final=r"./results_final", results_graphs=r"./results_graphs", splits=10, epochs=10):
    metrics = ['accuracy', 'precision', 'recall', 'fscore', 'loss']
    val_metrics = ['val_accuracy', 'val_precision',
                   'val_recall', 'val_fscore', 'val_loss']
    csv_list, csv_names, _ = ut.get_csvs_paths(results_final)
    history = []
    for csv in csv_list:
        history.append(pd.read_csv(csv))

    for m, vm in zip(metrics, val_metrics):
        metric_result_list = []
        val_metric_result_list = []
        model_result_names = []
        val_model_result_names = []
        for i, (h, name) in enumerate(zip(history, csv_names)):
            folds = np.array([h.pop(m)]).reshape([splits, epochs])
            val_folds = np.array([h.pop(vm)]).reshape([splits, epochs])
            metric_result_list.append(np.mean(folds, axis=0))
            val_metric_result_list.append(np.mean(val_folds, axis=0))
            model_result_names.append(name)
        display_graph(metric_result_list, val_metric_result_list,
                      model_result_names, m, results_graphs)
