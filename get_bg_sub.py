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
import lk
#import cnn


def get_bg_sub():
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
    valid, video, frame = util.load_video(vid_names[0])
    comp_img = frame[40:680, 70:1210, :]

    height = 680
    width = 1140

    x_range = range(140)
    y_range = range(80)

    best_mse = float('inf')
    best_start_locs = [0, 0]

    test_img = bg_frame[32: 672, 60: 1200]

    comp_img = cv2.cvtColor(comp_img, cv2.COLOR_RGB2GRAY)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

    diff = np.abs((comp_img / 255.) - (test_img / 255.))
    mask = np.zeros_like(diff)
    thresh = 0.1

    mask[diff <= thresh] = 0
    mask[diff > thresh] = 1

    frame = cv2.bitwise_and(comp_img, comp_img, mask=mask.astype(np.uint8))

    cv2.imshow('bg subtract', frame)
    cv2.waitKey(0)

    # for i in y_range:
    #     for j in x_range:
    #         test_img = bg_frame[i: i + 640, j: j + 1140, :]
    #
    #         mask = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    #         img = cv2.cvtColor(comp_img.copy(), cv2.COLOR_RGB2GRAY)
    #         bg_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    #
    #         if frame.shape == bg_frame.shape:
    #             diff = np.abs((img / 255.) - (bg_img / 255.))
    #         else:
    #             # create mask based on thresholded correlation between frames
    #             diff = cv2.filter2D(img, ddepth=-1, kernel=bg_img)
    #
    #         mse = np.sqrt(np.sum(np.square(diff)))
    #         if mse < best_mse:
    #             best_mse = mse
    #             best_start_locs = [i, j]
    #
    # print("Best min:", best_mse, " best img bounds:", best_start_locs)


if __name__ == '__main__':
    get_bg_sub()