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


def load_video(vid_name):
    """
    Load new video and return cv2 Video object and initial frame.

    :param vid_name:
    :return:
    """

    # init output vars
    valid = False
    # noinspection PyArgumentList
    video = cv2.VideoCapture(vid_name)
    init_frame = np.array([], dtype=np.uint8)

    # try load video and print out any issues
    if video is None or not video.isOpened():
        print("Video not found!")
    else:
        valid, init_frame = video.read()

        if valid is False or valid is None:
            print("Error extracting frame from video!")

    return valid, video, init_frame


def extract_bg_image(vid_name):
    """
    Extract background image from background video.

    :param vid_name:
    :return:
    """

    valid, video, init_frame = load_video(vid_name)

    # TODO: Just to ensure it's working properly
    cv2.imshow('background image', init_frame)
    cv2.waitKey(0)

    if valid:
        return init_frame


def process_video(vid_name, bg_frame):
    """
    Parse video for the relevant motion/lack-of-motion detection.

    :param dir_name:
    :return:
    """

    valid, video, frame = load_video(vid_name)

    while valid:
        diff_frame = background_subtraction(frame, bg_frame, thresh=0.2)

        #TODO: Finish


def background_subtraction(frame, bg_frame, thresh=0.2):
    """
    Uses background subtraction to subtract out stationarity in frame diffs.

    :param frame:
    :param bg_frame:
    :return:
    """

    #TODO: May not work without exact bg frame

    mask = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)
    bg_img = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2GRAY)

    if frame.shape == bg_frame.shape:
        diff = np.abs((img / 255.) - (bg_img / 255.))
    else:
        # create mask based on thresholded correlation between frames
        diff = cv2.filter2D(img, ddepth=-1, kernel=bg_img)

    mask[diff > thresh] = 0
    mask[diff <= thresh] = 1

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frame


if __name__ == '__main__':
    print("Please run the 'main.py' file.")
