import argparse
import os

import movenet as mn
import vid2pic as v2p
import train as t

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./video')
parser.add_argument('--pic', type=str, default='./input')
# parser.add_argument('--input', type=str, default='./input')
parser.add_argument('--csv', type=str, default='./output')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--final-results', type=str, default='./final_results')
parser.add_argument('--model', type=str, default='li')
# parser.add_argument('--n_images', type=int, default=0)
# parser.add_argument("--pose", type=int, default=0)  # 0-straight, 1-slouche
parser.add_argument('--skip-vid2pic', type=int, default=1)
parser.add_argument('--skip-movenet', type=int, default=1)
parser.add_argument('--skip-learning', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


def main():
    csvs = []

    if not args.skip_vid2pic:
        input_vid_names, pic_dir_paths, pose_type = v2p.vid2pic(
            args.video, args.pic)
    else:
        print("Skipping vid2pic segmentation")

    if not os.path.exists(args.csv):
        os.makedirs(args.csv)

    if not args.skip_movenet:
        csvs = mn.movenet(pic_dir_paths, input_vid_names,
                          args.csv, pose_type, args.model)
    else:
        csvs = None
        print("skipping movenet")

    if not os.path.exists(args.results):
        os.makedirs(args.results)
    if not os.path.exists(args.csv):
        os.makedirs(args.csv)

    t.train(csvs, args.csv, args.results, args.final_results, args.epochs)

    print("STOP")


main()
