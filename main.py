import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import argparse
import os
from tqdm import tqdm

# Import matplotlib libraries
from matplotlib import pyplot as plt

import util
import drawing_util as du
import cropping as cr


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='li')
parser.add_argument('--input', type=str, default='./input')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--n_images', type=int, default=100)
args = parser.parse_args()


# @param ["movenet_thunder", "movenet_lightning"]
model_name = "movenet_lightning"

if (model_name == "movenet_lightning" or model_name == 'li'):
    module = tf.saved_model.load(
        "./models/lightning")
    input_size = 192
elif (model_name == "movenet_thunder" or model_name == 'th'):
    module = tf.saved_model.load("./models/thunder")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)

movenet = module.signatures['serving_default']

n_images = args.n_images
filenames = [
    f.path for f in os.scandir(args.input) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
if len(filenames) > n_images:
    filenames = filenames[:n_images]

initial_image = tf.io.read_file(filenames[0])
initial_image = tf.image.decode_jpeg(initial_image)

n_images = len(filenames)
image_height, image_width, _ = initial_image.shape
crop_region = cr.init_crop_region(image_height, image_width)

output_images = []
start = time.time()

for fname in tqdm(filenames, desc="FILE", ascii=True, total=n_images):
    # Load the input image.

    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image)
    keypoints_with_scores = cr.run_inference(
        movenet, image[:, :, :], crop_region,
        crop_size=[input_size, input_size])
    output_images.append(du.draw_prediction_on_image(
        image[:, :, :].numpy().astype(np.int32),
        keypoints_with_scores, crop_region=None,
        close_figure=True, output_image_height=image_height))
    crop_region = cr.determine_crop_region(
        keypoints_with_scores, image_height, image_width)

print(': : : Average FPS:', n_images / (time.time() - start))

output = np.stack(output_images, axis=0)
du.to_gif(output, fps=10)
