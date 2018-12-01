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
from custom_lk import CustomLK


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
    # cv2.imshow('background image', init_frame)
    # cv2.waitKey(0)

    if valid:
        return init_frame


def process_frame(init_frame, next_frame, bg_frame, custom_lk, haar_classifier):
    """
    Parse video for the relevant motion/lack-of-motion detection.

    :param dir_name:
    :return:
    """

    init_frame = init_frame[40: 680, 70: 1210]

    orig_next_frame = next_frame.copy()
    next_frame = next_frame[40: 680, 70: 1210]

    # rescale to gray
    if len(init_frame.shape) > 2:
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    if len(next_frame.shape) > 2:
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    _, dilated_mask = background_subtraction(init_frame, bg_frame, thresh=0.25)

    u, v, img, next_frame = custom_lk.hierarchical_lk(img_a=init_frame,
                                                      img_b=next_frame,
                                                      orig_b=orig_next_frame,
                                                      levels=5,
                                                      k_size=8,
                                                      k_type="uniform",
                                                      sigma=0,
                                                      interpolation=cv2.INTER_CUBIC,
                                                      border_mode=cv2.BORDER_REPLICATE,
                                                      mask=dilated_mask)

    _, boxes = haar_classifier.process_frame(orig_next_frame)

    for (x, y, w, h) in boxes:
        next_frame = cv2.rectangle(orig_next_frame, (x, y), (x + w, y + h),
                                  color=(0, 255, 0), thickness=2)

    return img, next_frame


def background_subtraction(frame, bg_frame, thresh=0.2,
                           target_size=(640, 1140)):
    """
    Uses background subtraction to subtract out stationarity in frame diffs.

    :param frame:
    :param bg_frame:
    :return:
    """

    if frame.shape[:2] != target_size:
        frame = frame[40: 680, 70: 1210]
    if bg_frame.shape[:2] != target_size:
        bg_frame = bg_frame[32: 672, 60: 1200]

    mask = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    if len(frame.shape) != 2:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        img = frame
    bg_img = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

    if img.shape == bg_img.shape:
        diff = np.abs((img / 255.) - (bg_img / 255.))
    else:
        # create mask based on thresholded correlation between frames
        diff = cv2.filter2D(img, ddepth=-1, kernel=bg_img)

    # threshold mask
    mask[diff <= thresh] = 0
    mask[diff > thresh] = 1

    # apply mask to frame and median filter
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = cv2.medianBlur(frame, ksize=3)

    # zero out empty areas and dilate mask
    mask[:140, :] = 0
    mask[520:, :] = 0
    mask[:, 150: 220] = 0
    mask[:, :100] = 0
    mask[:, 1000:] = 0
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated_mask = cv2.dilate(mask, kernel=elem)

    return frame, dilated_mask


if __name__ == '__main__':
    print("Please run the 'main.py' file.")
