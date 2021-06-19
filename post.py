#!/usr/bin/env python
import util as ut
import numpy as np
from scipy import stats
import latextabs as lt


def post():
    # Parameters
    used_test = stats.ttest_rel
    alpha = 0.05

    # Load results
    legend = ut.json2object("results/legend.json")
    models = legend["models"]
    models = [m.replace("_", "-") for m in models]
    metrics = legend["metrics"]
    folds = legend["folds"]
    rescube = np.load("results/rescube.npy")

    # storage for ranks
    ranks = np.zeros((len(metrics), len(models)))

    table_file = open("results/tab.tex", "w")
    table_file.write(lt.header4classifiers(models))
    # First generate tables for each metric
    for mid, metric in enumerate(metrics):

        # Subtable is 2d (clf, fold)
        # rescube : fold, model, metric
        subtable = rescube[:, :, mid].T
        print("subtable: ", subtable)

        # Check if metric was valid
        if np.isnan(subtable).any():
            print("Unvaild")
            continue

        # Scores as mean over folds
        scores = np.mean(subtable, axis=1)
        stds = np.std(subtable, axis=1)

        t_statistic = np.zeros((len(models), len(models)))
        p_value = np.zeros((len(models), len(models)))

        for i in range(len(models)):
            for j in range(len(models)):
                si = subtable[i]
                sj = subtable[j]
                t_statistic[i, j], p_value[i, j] = used_test(
                    subtable[i], subtable[j])

        print(t_statistic)
        print("p value:")
        print(p_value)

        advantage = np.zeros((len(models), len(models)))
        advantage[t_statistic > 0] = 1

        significance = np.zeros((len(models), len(models)))
        significance[p_value <= alpha] = 1

        stat_better = significance * advantage

        print(stat_better)
        print(scores)
        print(stds)
        table_file.write(lt.row(metric, scores, stds))
        table_file.write(lt.row_stats(metric, stat_better, scores, stds))

    table_file.write(lt.footer("Results for %s metric" % metric))
    table_file.close()
