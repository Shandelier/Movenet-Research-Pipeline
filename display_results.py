import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import numpy as np
import util as ut
import os

basic = ["#3772ff", "#df2935", "#fdca40", "#080708", "#2FCF51"]
basic_saturated = ["#3874ff", "#fb0e1e", "#ffcb3d", "#06f93a", "#0e010e"]
graph_frame = {
    'accuracy': [0.8, 0.975],
    'precision': [0.75, 1],
    'recall': [0.70, 1],
    'fscore': [0.8, .975],
    'loss': [0.04, 0.15],
    'sensitivity': [0.7, 1],
    'specificity': [0.75, 1],
    'gmean': [0.8, .975],
    'bac': [0.80, .975],
}


def display_graph(results, val_results, stds, val_stds, model_names, metric_name, results_graphs):
    plt.figure(figsize=(8, 6))

    max_epoch = results[0].size
    for r in results:
        if max_epoch < r.size:
            max_epoch = r.size

    # epochs = np.linspace(0, max_epoch, max_epoch)
    fig, ax = plt.subplots()
    for i, (result, val_result, std, val_std, model_name, color) in enumerate(zip(results, val_results, stds, val_stds, model_names, basic_saturated)):
        # print(result)
        # diff = len(epochs) - result.size
        # if diff != 0:
        #     zero = np.zeros(diff)
        #     array = result.values.tolist()
        #     array =
        #     result = result.append(zero)
        epochs = np.linspace(1, result.size, result.size)

        # plt.plot(epochs,
        #          result, label=model_name, color=color, linestyle=':',)
        # plt.plot(epochs, std, color=color)

        plt.errorbar(epochs, result, yerr=std, color=color,
                     label="training_"+model_name, fmt='.', capsize=3, capthick=0.5)
        plt.errorbar(epochs, val_result, yerr=val_std, color=color,
                     label="validation_"+model_name, linestyle='solid', capsize=3, capthick=0.5)

        # plt.errorbar(epochs,
        #              val_result, yerr=val_std, label="val_"+model_name, color=color, linestyle='solid')
        plt.xlabel('Epoch')
        plt.ylim(graph_frame.get(metric_name))
        plt.ylabel(metric_name)
        plt.legend()
    plt.savefig(os.path.join(results_graphs,
                             '{}.png'.format(metric_name)), dpi=300)


def disp(results_final=r"./results_final", results_graphs=r"./results_graphs", splits=10, epochs=10):
    metrics = ['accuracy', 'bac', 'precision', 'recall',
               'fscore', 'loss', 'specificity', 'sensitivity', 'gmean']

    val_metrics = ['val_'+m for m in metrics]
    csv_list, csv_names, _ = ut.get_csvs_paths(results_final)

    history = []
    for csv in csv_list:
        history.append(pd.read_csv(csv))

    for m, vm in zip(metrics, val_metrics):
        metric_result_list = []
        metric_std_list = []
        val_metric_result_list = []
        val_metric_std_list = []
        model_result_names = []
        for i, (h, name) in enumerate(zip(history, csv_names)):
            folds = np.array([h.pop(m)]).reshape([splits, epochs])
            val_folds = np.array([h.pop(vm)]).reshape([splits, epochs])
            # skl_metric_result_list = np.array([skl_h.pop(m)])

            metric_result_list.append(np.mean(folds, axis=0))
            metric_std_list.append(np.std(folds, axis=0))

            val_metric_result_list.append(np.mean(val_folds, axis=0))
            val_metric_std_list.append(np.std(val_folds, axis=0))

            model_result_names.append(name)
        display_graph(metric_result_list, val_metric_result_list, metric_std_list, val_metric_std_list,
                      model_result_names, m, results_graphs)
    print("Graphs ready")


disp(epochs=50)
