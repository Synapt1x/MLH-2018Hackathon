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

    valid, video, frame = util.load_video(vid_names[0])
    init_frame = frame[40: 680, 70: 1210]

    valid, next_frame = video.read()
    orig_next_frame = next_frame.copy()
    next_frame = next_frame[40: 680, 70: 1210]

    # rescale to gray
    if len(init_frame.shape) > 2:
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    if len(next_frame.shape) > 2:
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    lk = LK()

    # lk.process_frame(init_frame, next_frame, bg_file=bg_file)

    custom_lk = CustomLK()

    u, v = custom_lk.hierarchical_lk(img_a=init_frame,
                                     img_b=next_frame,
                                     levels=5,
                                     k_size=8,
                                     k_type="uniform",
                                     sigma=0,
                                     interpolation=cv2.INTER_CUBIC,
                                     border_mode=cv2.BORDER_REPLICATE)
    print("u:", u, "v:", v)

    u = u / np.max(u)
    v = v / np.max(v)

    img = custom_lk.quiver(u, v, scale=100, stride=10)
    img = cv2.add(orig_next_frame[40:680, 70: 1210], img)
    cv2.imshow('img.png', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_lk()