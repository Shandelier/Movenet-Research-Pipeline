import tensorflow as tf
import os


def choose_model(model_name="movenet_lightning"):
    if (model_name == "movenet_lightning" or model_name == 'li'):
        module = tf.saved_model.load(
            "./models/lightning")
        input_size = 192
    elif (model_name == "movenet_thunder" or model_name == 'th'):
        module = tf.saved_model.load("./models/thunder")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    return module.signatures['serving_default'], input_size


def get_filenames(n_images, input):
    filenames = [
        f.path for f in os.scandir(input) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    if n_images == 0:
        return filenames
    if len(filenames) > n_images:
        filenames = filenames[:n_images]
    return filenames


def get_vid_paths_and_names(input):
    file_paths = [
        f.path for f in os.scandir(input) if f.is_file() and f.path.endswith(('.MOV', '.mp4'))]
    file_names = [os.path.basename(f).split(".", 1)[0] for f in file_paths]
    pose_type = [f.split("_", 1)[0] for f in file_names]

    return file_paths, file_names, pose_type
