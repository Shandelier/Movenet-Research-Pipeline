import numpy as np
import pandas as pd
import os
import argparse
import util as ut
# import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default=r'./output')
parser.add_argument('--results', type=str, default=r'./normalized_output')
args = parser.parse_args()


def main():
    # read csv
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    csvs, file_names, _ = ut.get_csvs_paths(args.csv)
    for i, (csv, file_name) in enumerate(zip(csvs, file_names)):
        ds = pd.read_csv(csv)
        n_samples = ds.shape
        if (n_samples[0] <= 0):
            raise ValueError(
                'the file {} does not contain any samples.'.format(csv))

        fp = ds.pop('filepath').to_numpy().reshape(n_samples[0], 1)
        pose_type = ds.pop('pose_type').to_numpy().reshape(n_samples[0], 1)
        points = ds.to_numpy().reshape([n_samples[0], 3, 17])
        points_backup = np.copy(points)

        xmin, ymin = np.amin(points[:, 1, :], 1), np.amin(points[:, 0, :], 1)

        points[:, 1, :] = points[:, 1, :] - xmin[:, None]
        points[:, 0, :] = points[:, 0, :] - ymin[:, None]

        xmax, ymax = np.amax(points[:, 1, :], 1), np.amax(points[:, 0, :], 1)

        points[:, 1, :] = points[:, 1, :]/xmax[:, None]
        points[:, 0, :] = points[:, 0, :]/ymax[:, None]

        xmax, xmin, ymax, ymin = np.copy(xmax).reshape(n_samples[0], 1), np.copy(xmin).reshape(
            n_samples[0], 1), np.copy(ymax).reshape(n_samples[0], 1), np.copy(ymin).reshape(n_samples[0], 1)

        points = points.reshape([n_samples[0], 51])

        write = np.concatenate((fp, pose_type, xmax, xmin,
                                ymax, ymin, points), axis=1).reshape([n_samples[0], 57])

        # do sprawdzania czy operacje są odwrotne i nie ma strat na dokładności floatów
        # points[:, 1, :] = (points[:, 1, :]*xmax) + xmin
        # points[:, 0, :] = (points[:, 0, :]*ymax) + ymin
        # points_backup = points_backup.reshape([n_samples[0], 51])
        # points = points.reshape([n_samples[0], 51])
        # diff = np.subtract(points_backup, points)
        path = os.path.join(args.results, file_name+".csv")
        np.savetxt(path,  write, delimiter=',')

    # normalize
    # save as new
    print("yo")


main()
