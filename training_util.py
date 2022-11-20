import tensorflow as tf
# import tensorflow_model_analysis as tfma
from tensorflow.keras import layers, regularizers
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score


def get_models_and_names(first_layer):
    models = []
    model_names = []

    model_names.append("1_layer_1024_dropout_05")
    model = tf.keras.Sequential([
        layers.Dense(first_layer),
        layers.Dense(1024, kernel_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("1_layer_1024")
    model = tf.keras.Sequential([
        layers.Dense(first_layer),
        layers.Dense(1024),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=METRICS)
    models.append(
        model
    )

    model_names.append("128_relu_dropout_50")
    model = tf.keras.Sequential([
        layers.Dense(first_layer),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l1(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l1(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l1(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l1(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=METRICS)
    models.append(
        model
    )

    # TODO: Droput monte carlo (tutorial robienia własnej warstywy)
    # y_samples = np.stack([model(test_rescale,training=True) for sample in range(100)])

    return models, model_names


class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.specificity = self.add_weight(
            name='specificity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        self.specificity.assign((tn) / (fp + tn + 1e-6))

    def result(self):
        return self.specificity

    def reset_state(self):
        self.tn.reset_state()
        self.fp.reset_state()
        self.specificity.assign(0)


class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.sensitivity = self.add_weight(
            name='sensitivity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.sensitivity.assign((tp) / (tp + fn + 1e-6))

    def result(self):
        return self.sensitivity

    def reset_state(self):
        self.tp.reset_state()
        self.fn.reset_state()
        self.sensitivity.assign(0.0)


class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='fscore', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='fscore', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_state(self):
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


class GeometricMean(tf.keras.metrics.Metric):
    def __init__(self, name='gmean', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.gmean = self.add_weight(name='gmean', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        spec = ((tn) / (fp + tn + 1e-6))
        sen = ((tp) / (tp + fn + 1e-6))
        self.gmean.assign(tf.math.sqrt(spec * sen))

    def result(self):
        return self.gmean

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.gmean.assign(0)


class Kappa(tf.keras.metrics.Metric):
    def __init__(self, name='kappa', **kwargs):
        super().__init__(name=name, **kwargs)
        # TODO: kappa doesn't work
        self.kappa_fn = cohen_kappa_score
        self.kappa = self.add_weight(name='kappa', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.kappa.assign(self.kappa_fn(y_true, y_pred))

    def result(self):
        return self.kappa

    def reset_state(self):
        self.kappa.assign(0)


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='bac', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.bac = self.add_weight(name='bac', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        spec = ((tn) / (fp + tn + 1e-6))
        sen = ((tp) / (tp + fn + 1e-6))
        self.bac.assign((sen + spec)/2)

    def result(self):
        return self.bac

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bac.assign(0)


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    BalancedAccuracy(),
    # Kappa(),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    F1_Score(),
    Specificity(),
    Sensitivity(),
    GeometricMean(),
    #    tf.keras.metrics.TruePositives(name='tp'),
    #    tf.keras.metrics.FalsePositives(name='fp'),
    #    tf.keras.metrics.TrueNegatives(name='tn'),
    #    tf.keras.metrics.FalseNegatives(name='fn'),
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
