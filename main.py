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
import movenet as mn
import vid2pic as v2p

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./video')
parser.add_argument('--pic', type=str, default='./input')
# parser.add_argument('--input', type=str, default='./input')
parser.add_argument('--csv', type=str, default='./output')
parser.add_argument('--model', type=str, default='li')
# parser.add_argument('--n_images', type=int, default=0)
# parser.add_argument("--pose", type=int, default=0)  # 0-straight, 1-slouche
args = parser.parse_args()


def main():
    input_vid_names, pic_dir_paths, pose_type = v2p.vid2pic(
        args.video, args.pic)
    mn.movenet(pic_dir_paths, input_vid_names,
               args.csv, pose_type, args.model)
    print("STOP")


main()
