import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import numpy as np


def display_graph(csvs, model_names, metric_name):
    plt.figure(figsize=(8, 6))

    for n, (csv, name) in enumerate(zip(csvs, model_names)):
        history = pd.read_csv(csv)
        epochs = np.linspace(0, len(history)-1, len(history))
        # plt.subplot(2, 2, n+1)
        plt.plot(epochs,
                 history, label=name)

        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        # if metric == 'loss':
        #     plt.ylim([0, plt.ylim()[1]])
        # elif metric == 'auc':
        #     plt.ylim([0.8, 1])
        # else:
        plt.ylim([0, 1])
        plt.legend()


def main():
    # accs, acc_model_names = ut.get_result_type('accuracy')
    metrics = ['accuracy', 'precision', 'recall', 'loss']
    for m in metrics:
        data, model_name = ut.get_result_type(m)
        display_graph(data, model_name, m)
    # pres, pre_model_names = ut.get_result_type('precision')


main()
