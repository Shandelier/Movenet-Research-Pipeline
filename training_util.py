import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_model_analysis as tfma
from tensorflow.keras import layers, regularizers
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score


def get_models_and_names():
    models = []
    model_names = []

    model_names.append("1_layer_1024_dropout_05")
    model = tf.keras.Sequential([
        layers.Dense(51),
        layers.Dense(1024, kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("1_layer_1024")
    model = tf.keras.Sequential([
        layers.Dense(51),
        layers.Dense(1024),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("128_relu_dropout_50")
    model = tf.keras.Sequential([
        layers.Dense(51),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(
        model
    )

    # TODO: Droput monte carlo (tutorial robienia własnej warstywy)
    # y_samples = np.stack([model(test_rescale,training=True) for sample in range(100)])

    return models, model_names


METRICS = [tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.TruePositives(name='tp'),
           tf.keras.metrics.FalsePositives(name='fp'),
           tf.keras.metrics.TrueNegatives(name='tn'),
           tf.keras.metrics.FalseNegatives(name='fn'),
           ]

skl_metrics = {
    "accuracy": accuracy_score,
    "bac": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "fscore": f1_score,
    "gmean": geometric_mean_score,
    "kappa": cohen_kappa_score,
}


excessive_pred = [
    'pred_nose',
    'pred_left_eye',
    'pred_right_eye',
    'pred_left_ear',
    'pred_right_ear',
    'pred_left_shoulder',
    'pred_right_shoulder',
    'pred_left_elbow',
    'pred_right_elbow',
    'pred_left_wrist',
    'pred_right_wrist',
    'pred_left_hip',
    'pred_right_hip',
    'pred_left_knee',
    'pred_right_knee',
    'pred_left_ankle',
    'pred_right_ankle']

excessive = [
    'x_left_hip',
    'x_right_hip',
    'x_left_knee',
    'x_right_knee',
    'x_left_ankle',
    'x_right_ankle',
    'y_left_hip',
    'y_right_hip',
    'y_left_knee',
    'y_right_knee',
    'y_left_ankle',
    'y_right_ankle']
