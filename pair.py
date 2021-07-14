from scipy.stats import ttest_rel
import numpy as np

scores = np.array(([[0.77697842, 0.76258993, 0.79710145, 0.80291971, 0.81751825, 0.75539568, 0.81294964, 0.76086957, 0.81751825, 0.78832117],
                    [0.67625899, 0.69064748, 0.70289855, 0.76642336, 0.69343066,
                        0.68345324, 0.67625899, 0.70289855, 0.72992701, 0.75912409],
                    [0.79856115, 0.83453237, 0.81884058, 0.83941606, 0.75182482, 0.76978417, 0.79856115, 0.8115942,  0.86131387, 0.81751825]]))

alfa = .05
clfs = [3, 3, 3]
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
