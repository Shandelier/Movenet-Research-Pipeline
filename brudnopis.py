# from sklearn.utils import shuffle
# import util as ut
# import os
# import pandas as pd
# import training_util as tut
# import numpy as np


# def sample_2000():
#     file_paths, file_names, pose_type = ut.get_csvs_paths(
#         os.path.join("5-people-csvs"))

#     init = list.pop(file_paths)
#     ds = pd.read_csv(init)
#     for i, csv in enumerate(file_paths):
#         read = pd.read_csv(csv)
#         ds = pd.concat([ds, read], axis=0)

#     # # ds.pop("filepath")
#     # for p in tut.excessive_pred:
#     #     ds.pop(p)
#     # for e in tut.excessive:
#     #     ds.pop(e)

#     ds = ds.sample(2000)
#     ds = shuffle(ds, random_state=420)
#     ds = ds.reset_index(drop=True)
#     print(ds.head())

#     ds.to_csv(os.path.join(
#         "results", "5_people_small.csv"), sep='\t', index=False, header=True)


# def negative():
#     ds = pd.read_csv(os.path.join(
#         "results", "5_people_small.csv"))
#     pose = ds.pop("pose_type")
#     file = ds.pop("filepath")

#     negative = ds.loc[df.value]
#     return 0


# sample_2000()


import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflowjs as tfjs
import os
import train as t
import util as ut
import training_util as tut


def main():
    model = tf.keras.Sequential([
        layers.Dense(22),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam())

    csvs, _, _ = ut.get_csvs_paths(r"./output")
    X, y, _ = t.load_split(csvs, 2, 2)

    model.fit(X, y, epochs=10)

    tfjs.converters.save_keras_model(model, tfjs_target_dir)


main()
