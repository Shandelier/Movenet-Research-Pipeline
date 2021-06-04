import os
import subprocess as sub
import util as ut


def vid2pic(input, output):
    # get input folder name, and sample class to create output dirs
    input_file_paths, input_file_names, pose_type = ut.get_vid_paths_and_names(
        input)

    output_pic_dir_path = [os.path.join(output, p) for p in input_file_names]
    for p in output_pic_dir_path:
        if not os.path.exists(p):
            os.makedirs(p)

    for i, (ip, op) in enumerate(zip(input_file_paths, output_pic_dir_path)):
        # ffmpeg -qmin 0 -qmax 1 are the best quality parameters
        print("Segment to pictures ", ip)
        out = os.path.join(op, "%07d.jpg")
        bash_cmd = ["ffmpeg", "-i", ip, "-qmin", "0", "-qmax",
                    "1", "-q:v", "1", "-async", "1", out]
        process = sub.Popen(bash_cmd)
        # save bash command output and errorlog
        outlog, error = process.communicate()

    return input_file_names, output_pic_dir_path, pose_type
