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
from custom_lk import CustomLK
from face_detector import FaceDetector
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

    valid, video, init_frame = util.load_video(vid_names[1])

    valid, next_frame = video.read()

    bg_sub_img, mask = util.background_subtraction(init_frame, bg_frame,
                                               thresh=0.25)

    # lk = LK()
    # lk.process_frame(init_frame, next_frame, bg_file=bg_file)

    custom_lk = CustomLK()
    haar_cascade = FaceDetector('body')

    writer = cv2.VideoWriter('output.avi', -1, 20, (1140, 640))

    frame_num = 1

    history = None

    while valid:

        img, frame, event, history = util.process_frame(init_frame, next_frame,
                                                        mask, custom_lk,
                                                        haar_cascade, history)

        init_frame = next_frame.copy()
        valid, next_frame = video.read()

        writer.write(img)

        cv2.imshow('image', img)
        cv2.waitKey(2)

        frame_num += 1

    writer.release()


if __name__ == '__main__':
    test_lk()
