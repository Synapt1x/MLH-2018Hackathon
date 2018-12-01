# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

main.py: this contains the main code for running the application.
"""

##############################################################################
__author__ = ["Chris Cadonic", "Cassandra Aldaba"]
__credits__ = ["Chris Cadonic", "Cassandra Aldaba"]
__version__ = "0.1"
##############################################################################


import os

import yaml
import util
import numpy as np
import cv2
from lk import LK
from custom_lk import CustomLK
#import cnn


def test_lk():
    """
    Main function for running the application.

    :return:
    """

    # get current directory to work relative to current file path
    curdir = os.path.dirname(__file__)

    # Load configuration for system
    yaml_file = os.path.join(curdir, 'config.yaml')
    with open(yaml_file, "r") as f:
        config = yaml.load(f)

    # extract list of videos from data dir
    vid_dir = os.path.join(curdir, config['traindir'])
    vid_names = util.load_data(vid_dir)

    # extract background subtraction image from bg vid
    bg_file = os.path.join(curdir, config['bg_img'])
    bg_valid, bg_video, bg_frame = util.load_video(bg_file)

    valid, video, frame = util.load_video(vid_names[1])
    init_frame = frame[40: 680, 70: 1210]

    valid, next_frame = video.read()
    orig_next_frame = next_frame.copy()
    next_frame = next_frame[40: 680, 70: 1210]

    # rescale to gray
    if len(init_frame.shape) > 2:
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    if len(next_frame.shape) > 2:
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    _, mask = util.background_subtraction(init_frame, bg_frame, thresh=0.25)
    mask[:140, :] = 0
    mask[520:, :] = 0
    mask[:, 150: 220] = 0
    mask[:, :100] = 0
    mask[:, 1000:] = 0
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated_mask = cv2.dilate(mask, kernel=elem)

    # lk = LK()
    # lk.process_frame(init_frame, next_frame, bg_file=bg_file)

    custom_lk = CustomLK()

    writer = cv2.VideoWriter('output.avi', -1, 20, (1140, 640))

    frame_num = 1

    while valid:

        print("Frame:", frame_num)

        u, v = custom_lk.hierarchical_lk(img_a=init_frame,
                                         img_b=next_frame,
                                         levels=4,
                                         k_size=12,
                                         k_type="uniform",
                                         sigma=0,
                                         interpolation=cv2.INTER_CUBIC,
                                         border_mode=cv2.BORDER_REPLICATE)

        u = u / np.max(u)
        v = v / np.max(v)

        dilated_mask *= 255
        u *= dilated_mask / 255.
        v *= dilated_mask / 255.

        img = custom_lk.quiver(u, v, scale=75, stride=10)
        img = cv2.add(orig_next_frame[40:680, 70: 1210], img)

        # cv2.imshow('img.png', img)
        # cv2.waitKey(10)

        writer.write(img)

        init_frame = next_frame.copy()
        valid, next_frame = video.read()
        orig_next_frame = next_frame.copy()
        next_frame = next_frame[40: 680, 70: 1210]
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        frame_num += 1

    writer.release()


if __name__ == '__main__':
    test_lk()
