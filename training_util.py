import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_model_analysis as tfma
from tensorflow.keras import layers


def get_models_and_names():
    models = []
    model_names = []

    model_names.append("model_throttle")
    model = tf.keras.Sequential([
        layers.Dense(22),
        layers.Dense(64),
        layers.Dense(128),
        layers.Dense(32),
        layers.Dense(128),
        layers.Dense(32),
        layers.Dense(8),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("model_1_layer_128")
    model = tf.keras.Sequential([
        layers.Dense(22),
        layers.Dense(128),
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(16),
        layers.Dense(8),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("model_128_relu")
    model = tf.keras.Sequential([
        layers.Dense(22),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=METRICS)
    models.append(model)

    # TODO: Droput monte carlo (tutorial robienia własnej warstywy)
    # y_samples = np.stack([model(test_rescale,training=True) for sample in range(100)])

    return models, model_names


METRICS = [tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.Accuracy(name='accuracy'),
           #    tfa.metrics.GeometricMean(name='gmean'),
           tf.keras.metrics.TruePositives(name='tp'),
           tf.keras.metrics.FalsePositives(name='fp'),
           tf.keras.metrics.TrueNegatives(name='tn'),
           tf.keras.metrics.FalseNegatives(name='fn'), ]


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
