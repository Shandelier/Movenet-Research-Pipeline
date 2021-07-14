

import training_util as tut
import pandas as pd
import os
import util as ut
from sklearn.utils import shuffle
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition,
                     random_projection)
print(__doc__)


file_paths, file_names, pose_type = ut.get_csvs_paths(
    os.path.join("5-people-csvs"))

init = list.pop(file_paths)
ds = pd.read_csv(init)
for i, csv in enumerate(file_paths):
    read = pd.read_csv(csv)
    ds = pd.concat([ds, read], axis=0)

ds.pop('filepath')
for p in tut.excessive_pred:
    ds.pop(p)
for e in tut.excessive:
    ds.pop(e)

ds = shuffle(ds, random_state=420)
ds = ds.reset_index(drop=True)
ds = ds.sample(n=200)
label = ds.pop("pose_type")
print(label)


X = ds.to_numpy()
y = label.to_numpy()
n_samples, n_features = X.shape
n_neighbors = 30


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 2.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")
plt.show()

# ----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding learning pca 50*1")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=420, perplexity=30,
                     learning_rate=1.0, n_iter=5000, metric='euclidean')
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

print("Computing t-SNE embedding learning pca 50*10")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=420, perplexity=30,
                     learning_rate=10.0, n_iter=5000, metric='euclidean')
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

print("Computing t-SNE embedding learning pca emtric 5*1")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=420, perplexity=5,
                     learning_rate=1.0, n_iter=5000, metric='euclidean')
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

print("Computing t-SNE embedding learning pca emtric 5*10")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=420, perplexity=5,
                     learning_rate=10.0, n_iter=5000, metric='euclidean')
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()
