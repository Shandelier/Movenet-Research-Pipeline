import argparse
import os
import glob

import movenet as mn
import vid2pic as v2p
import train as t
import display_results as dis
import post

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default=r'./video')
parser.add_argument('--pic', type=str, default=r'./input')
# parser.add_argument('--input', type=str, default='./input')
parser.add_argument('--csv', type=str, default=r'./output')
parser.add_argument('--results', type=str, default=r'./results')
parser.add_argument('--results-final', type=str, default=r'./results_final')
parser.add_argument('--results-graphs', type=str, default=r'./results_graphs')
parser.add_argument('--model', type=str, default='li')
# 0 - no, 1 - results, 2 - all dirs
parser.add_argument('--clear_dir', type=int, default=1)
# parser.add_argument('--n_images', type=int, default=0)
# parser.add_argument("--pose", type=int, default=0)  # 0-straight, 1-slouche
parser.add_argument('--skip-vid2pic', type=int, default=1)
parser.add_argument('--skip-movenet', type=int, default=1)
parser.add_argument('--skip-learning', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


def main():
    if args.clear_dir == 1:
        clear_dirs()

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
    if not os.path.exists(args.results_final):
        os.makedirs(args.results_final)
    if not os.path.exists(args.results_graphs):
        os.makedirs(args.results_graphs)

    splits = t.train(csvs, args.csv, args.results,
                     args.results_final, args.epochs)

    dis.disp(args.results_final, args.results_graphs, splits, args.epochs)

    post.post()

    print("STOP")


def clear_dirs():
    results_files = glob.glob(os.path.join(args.results, '*'))
    results_final_files = glob.glob(os.path.join(args.results_final, '*'))
    results_graphs_files = glob.glob(os.path.join(args.results_graphs, '*'))
    for f in results_files:
        os.remove(f)
    for f in results_final_files:
        os.remove(f)
    for f in results_graphs_files:
        os.remove(f)


main()
