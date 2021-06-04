import argparse
import os
import time
import subprocess as sub
from tqdm import tqdm
from datetime import datetime

import util as ut
import save_utils as su
import cropping as cr

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./video')
parser.add_argument('--output', type=str, default='./input')
args = parser.parse_args()


# get input folder name, and sample class to create output dirs
input_file_paths, input_file_names, pose_type = ut.get_vid_paths_and_names(
    args.input)

output_pic_dir_path = [os.path.join(args.output, p) for p in input_file_names]
for p in output_pic_dir_path:
    if not os.path.exists(p):
        os.makedirs(p)

# for p in input_file_paths:
# zajebanie komendy ffmpeg
bash_cmd = ["ls", "."]
process = sub.Popen(bash_cmd)
output, error = process.communicate()
print(output)
print(error)
