import numpy as np


def header4classifiers(classifiers):
    text = ""
    text += "\\begin{table}[!ht]\n"
    text += "\\centering\n"
    text += "\\scriptsize\n"
    text += "\\begin{tabular}{l|\n"
    for i, clf in enumerate(classifiers):
        text += "S[table-format=0.3, table-figures-uncertainty=1]%s\n" % (
            "|" if i != len(classifiers)-1 else "}"
        )
    text += "\\toprule"
    text += "\\bfseries Metric &\n"

    for i, clf in enumerate(classifiers):
        text += "\\multicolumn{1}{c%s}{\\bfseries %s} %s\n" % (
            "|" if i != len(classifiers)-1 else " ",
            clf + " (%i)" % (i+1),
            "&" if i != len(classifiers)-1 else "\\\\"
        )
    text += "\\midrule\n"
    """
    \toprule
      \bfseries Metric &
      \multicolumn{1}{c|}{\bfseries CLF1} &
      \multicolumn{1}{c }{\bfseries CLF2} \\
      \midrule
    """

    return text


def row(dataset, scores, stds):
    text = "\\emph{\\textbf{%s}}" % dataset
    for i in range(scores.shape[0]):
        text += "& "
        text += "%.3f(%i) " % (scores[i], int(stds[i]*1000))

    text += "\\\\\n"
    return text


def row_stats(dataset, dependency, scores, stds):
    text = "\\ "
    min_score = np.min(scores)
    for i in range(scores.shape[0]):
        text += "& "
        a = np.where(dependency[i] == 1)[0]
        # a = np.where(np.logical_and(dependency[i] == 0, scores[i]>min_score))[0]

        for value in a:
            sth1 = scores[i]
            sth2 = scores[value]
            if scores[i] < scores[value]:
                a = a[a != value]

        if a.size == scores.shape[0]-1:
            text += "$_{all}$"
        elif a.size == 0:
            text += "$_{-}$"
        else:
            a += 1
            text += "$_{" + ", ".join(["%i" % i for i in a]) + "}$"

    text += "\\\\\n"
    text += " & & & \\\\\n"
    return text


def footer(caption):
    text = ""
    text += "\\bottomrule\n"
    text += "\\end{tabular}\n"
    text += "\\caption{%s}\n" % caption
    text += "\\end{table}\n"
    return text
