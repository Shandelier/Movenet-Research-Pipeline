import tensorflow as tf
import numpy as np
import time
import os
from tqdm import tqdm

# Import matplotlib libraries

import util as ut
import save_utils as su
import cropping as cr


def movenet(pic_paths, pic_dir_names, output_path, pose_type, model_type="li"):
    # get input folder name to create output file name
    # input_dir_name = os.path.basename(pic_path).split("./", 1)[0]
    # start_date = datetime.now().strftime("--%H-%M--%d-%m-%Y")
    # output_pic_dir_name = args.output + "/" + input_dir_name + start_date
    output_csv_dir_names = [os.path.join(
        output_path, pdn)+".csv" for i, pdn in enumerate(pic_dir_names)]
    # output_csv_dir_name = os.path.join(output_path, pic_dir_name)+".csv"

    movenet, input_size = ut.choose_model(model_type)

    for i, (pp, ocsv, pose) in enumerate(zip(pic_paths, output_csv_dir_names, pose_type)):
        filenames = ut.get_filenames(0, pp)
        n_images = len(filenames)

        initial_image = tf.io.read_file(filenames[0])
        initial_image = tf.image.decode_jpeg(initial_image)
        image_height, image_width, _ = initial_image.shape
        crop_region = cr.init_crop_region(image_height, image_width)

        # output_images = []
        start = time.time()

        # PREPARE CSV FILE
        csv_file = open(ocsv, "ab")
        np.savetxt(csv_file, su.KEYPOINT_LABELS, delimiter=",", fmt="%s")

        for fname in tqdm(filenames, desc="FILE", ascii=True, total=n_images):
            # Load the input image.
            image = tf.io.read_file(fname)
            image = tf.image.decode_jpeg(image)
            keypoints_with_scores = cr.run_inference(
                movenet, image[:, :, :], crop_region,
                crop_size=[input_size, input_size])

            write = np.hstack([fname,
                               pose, np.squeeze(keypoints_with_scores)
                               .flatten()]).reshape([1, 53])
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
    return output_csv_dir_names
