import tensorflow as tf
import os
import glob
import json


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
    file_paths = [os.path.join(path, name) for path, subdirs,
                  files in os.walk(input) for name in files]
    file_names = [os.path.basename(f).split(".", 1)[0] for f in file_paths]
    pose_type = [f.split("_", 1)[0] for f in file_names]

    return file_paths, file_names, pose_type


def get_csvs_paths(input):
    file_paths = [
        f.path for f in os.scandir(input) if f.is_file() and f.path.endswith('.csv')]
    file_names = [os.path.basename(f).split(".", 1)[0] for f in file_paths]
    pose_type = [f.split("_", 1)[0] for f in file_names]

    return file_paths, file_names, pose_type


def get_pics_paths(input):
    file_paths = [
        f.path for f in os.scandir(input) if f.is_dir()]
    file_names = [os.path.basename(f).split(".", 1)[0] for f in file_paths]
    pose_type = [f.split("_", 1)[0] for f in file_names]

    return file_paths, file_names, pose_type


def get_paths(input, file_extension):
    file_paths = [os.path.join(path, name) for path, subdirs,
                  files in os.walk(input) for name in files]
    file_names = [os.path.basename(f).split(".", 1)[
        0] for f in file_paths]
    pose_type = [f.split("_", 1)[0] for f in file_names]

    return file_paths, file_names, pose_type


def get_result_type(metric_str, rdir='results'):
    path_pattern = os.path.join(rdir, metric_str+'*.csv')
    file_paths = glob.glob(path_pattern)
    file_names = [f.split("_", 1)[1] for f in file_paths]
    file_names = [f.split(".", 1)[0] for f in file_names]

    return file_paths, file_names


def json2object(path):
    with open(path) as json_data:
        return json.load(json_data)
