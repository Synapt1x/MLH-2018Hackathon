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
from face_detector import FaceDetector
#import cnn


def test_faceDetect():
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

    valid, video, frame = util.load_video(vid_names[1])
    frame = frame[32: 672, 60: 1200]

    faceDetector = FaceDetector()
    frame = faceDetector.process_frame(frame)

    while valid:
        valid, frame = video.read()
        frame = faceDetector.process_frame(frame)


if __name__ == '__main__':
    test_faceDetect()