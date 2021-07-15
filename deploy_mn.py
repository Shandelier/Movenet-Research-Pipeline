import tensorflow as tf
import tensorflow_hub as hub


def deploy_mn():

    module = tf.saved_model.load("./models/lightning")
    input_size = 192
    print(module.graph)
    print(module.variables)
    print(module.signatures)
    movenet = module.signatures['serving_default']


deploy_mn()
