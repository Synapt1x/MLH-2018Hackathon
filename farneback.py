# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

farneback.py: this contains the code for polynomial expansion optical flow.
"""

##############################################################################
__author__ = ["Chris Cadonic", "Cassandra Aldaba"]
__credits__ = ["Chris Cadonic", "Cassandra Aldaba"]
__version__ = "0.1"
##############################################################################


import os

import cv2
import numpy as np
import util


class Farneback:

    def __init__(self, config=None):

        if config is not None:
            self.config = config

    def process_frame(self, prev_frame, frame, bg_file=None):

        bg_valid, bg_video, bg_frame = util.load_video(bg_file)

        prev_frame = util.background_subtraction(prev_frame, bg_frame,
                                                 thresh=0.15)
        frame = util.background_subtraction(frame, bg_frame,
                                            thresh=0.15)

        cv2.imwrite('prev_frame.png', prev_frame)
        cv2.imwrite('frame.png', frame)

        # rescale to gray
        if len(prev_frame.shape) > 2:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # reshape to correct size
        if prev_frame.shape[:2] != (640, 1140):
            prev_frame = prev_frame[40: 680, 70: 1210]
        if frame.shape[:2] != (640, 1140):
            frame = frame[40: 680, 70: 1210]

        mask = np.zeros_like(frame, dtype=np.float)
        hsv = np.zeros(shape=(frame.shape[0], frame.shape[1], 3),
                       dtype=np.uint8)

        hsv[..., 1] = 255
        # hsv = np.expand_dims(hsv, axis=2)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        #final_frame = cv2.add(frame, mask.astype(np.uint8))

        cv2.imshow("moving in frame", rgb)
        cv2.waitKey(0)

        return frame, rgb


if __name__ == '__main__':
    print("Please run the 'main.py' file.")