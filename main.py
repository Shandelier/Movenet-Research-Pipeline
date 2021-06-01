import tensorflow as tf
import numpy as np
import time
import argparse
import os
from tqdm import tqdm

# Import matplotlib libraries
from datetime import datetime

import util as ut
import save_utils as su
import cropping as cr


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='li')
parser.add_argument('--input', type=str, default='./input')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--n_images', type=int, default=0)
parser.add_argument("--pose", type=int, default=0)  # 0-straight, 1-slouche
args = parser.parse_args()

if args.input == args.output:
    print(
        "[WARNING] input dir is the same as output dir -- the pictures will be overwritten"
    )
    print("Do you wish to continue?: y/n")
    if input() != "y":
        exit()

# get input folder name to create output file name
input_dir_name = os.path.basename(args.input).split("./", 1)[0]
start_date = datetime.now().strftime("--%H-%M--%d-%m-%Y")
output_pic_dir_name = args.output + "/" + input_dir_name + start_date
output_csv_dir_name = args.output + "/" + input_dir_name + start_date + ".csv"
if not os.path.exists(output_pic_dir_name):
    os.makedirs(output_pic_dir_name)


movenet, input_size = ut.choose_model(args.model)

# TODO: sort files according to system sorting not python
filenames = ut.get_filenames(args.n_images, args.input)
n_images = len(filenames)

initial_image = tf.io.read_file(filenames[0])
initial_image = tf.image.decode_jpeg(initial_image)
image_height, image_width, _ = initial_image.shape
crop_region = cr.init_crop_region(image_height, image_width)

# output_images = []
start = time.time()

# PREPARE CSV FILE
csv_file = open(output_csv_dir_name, "ab")
np.savetxt(csv_file, su.KEYPOINT_LABELS, delimiter=",", fmt="%s")

for fname in tqdm(filenames, desc="FILE", ascii=True, total=n_images):
    # Load the input image.
    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image)
    keypoints_with_scores = cr.run_inference(
        movenet, image[:, :, :], crop_region,
        crop_size=[input_size, input_size])

    write = np.hstack([fname,
                      args.pose,
                      np.squeeze(keypoints_with_scores).flatten()]).reshape([1, 53])
    np.savetxt(csv_file, write, delimiter=",", fmt='%s')

    # output_images.append(du.draw_prediction_on_image(
    #     image[:, :, :].numpy().astype(np.int32),
    #     keypoints_with_scores, crop_region=None,
    #     close_figure=True, output_image_height=image_height))
    crop_region = cr.determine_crop_region(
        keypoints_with_scores, image_height, image_width)

print(': : : Average FPS:', n_images / (time.time() - start))
csv_file.close()

# output = np.stack(output_images, axis=0)
# du.to_gif(output, fps=10)
