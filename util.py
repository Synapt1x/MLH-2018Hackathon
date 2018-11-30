# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

util.py: this contains code for processing and helper functions for the
application.
"""

##############################################################################
__author__ = ["Chris Cadonic", "Cassandra Aldaba"]
__credits__ = ["Chris Cadonic", "Cassandra Aldaba"]
__version__ = "0.1"
##############################################################################


import os

import glob
import cv2
import numpy as np


def load_data(dir_name=None):
    """
    Loads data from the specified directory 'dir_name'.

    :param dir_name: (String) - os path string denoting data directory
    :return:
    """

    # extract all files from directory
    output_files = os.listdir(dir_name)
    output_files = [os.path.join(dir_name, file) for file in output_files]

    return output_files


def background_subtraction(frame, bg_frame, thresh=0.2):
    """
    Uses background subtraction to subtract out stationarity in frame diffs.

    :param frame:
    :param bg_frame:
    :return:
    """

    #TODO: Finish/Verify
    diff = np.zeros(shape=frame.shape)
    if frame.shape == bg_frame.shape:
        diff = (frame.astype / 255.) - (bg_frame / 255.)
    else:
        # create mask based on thresholded correlation between frames
        diff = cv2.filter2D(frame, ddepth=-1, kernel=bg_frame)

    diff[diff > thresh] = 0
    diff[diff <= thresh] = 1

    frame = cv2.bitwise_and(frame, frame, mask=diff)

    return frame


if __name__ == '__main__':
    print("Please run the 'main.py' file.")
