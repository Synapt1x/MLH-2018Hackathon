# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

lk.py: this contains the code for hierarchical Lucas-Kanade optical flow.
"""

##############################################################################
__author__ = ["Chris Cadonic", "Cassandra Aldaba"]
__credits__ = ["Chris Cadonic", "Cassandra Aldaba"]
__version__ = "0.1"
##############################################################################


import os

import cv2
import numpy as np


class LK:

    def __init__(self, config=None):

        if config is not None:
            self.config = config

    def process_frame(self, prev_frame, frame):

        if len(prev_frame.shape) > 2:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        features = cv2.goodFeaturesToTrack(image=prev_frame,
                                           maxCorners=0,
                                           qualityLevel=1.0,
                                           minDistance=1.0)
        output = cv2.calcOpticalFlowPyrLK(prev_frame, frame, features)



if __name__ == '__main__':
    print("Please run the 'main.py' file.")