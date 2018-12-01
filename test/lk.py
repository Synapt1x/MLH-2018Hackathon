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
import util


class LK:

    def __init__(self, config=None):

        if config is not None:
            self.config = config

    def process_frame(self, prev_frame, frame, features=None, bg_file=None):

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

        # if features not passed in; extract good features
        if features is None:
            features = cv2.goodFeaturesToTrack(image=prev_frame,
                                               maxCorners=100,
                                               qualityLevel=0.3,
                                               minDistance=7,
                                               blockSize=7)

        new_features, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                     features,
                                          None,
                                          winSize=(21, 21),
                                          maxLevel=5,
                                          criteria=(
                                          cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                          10, 0.03))

        new_locs = new_features[st == 1].astype(np.int)
        old_locs = features[st == 1].astype(np.int)

        for i, (new_pnt, old_pnt) in enumerate(zip(new_locs, old_locs)):
            new_pnt = tuple(new_pnt)
            old_pnt = tuple(old_pnt)
            cv2.line(mask, new_pnt, old_pnt, color=(0, 255, 0), thickness=2)
            cv2.circle(mask, new_pnt, color=(255, 0, 0), radius=5, thickness=2)

        final_frame = cv2.add(frame, mask.astype(np.uint8))

        cv2.imshow("moving in frame", final_frame)
        cv2.waitKey(0)

        return frame, final_frame


if __name__ == '__main__':
    print("Please run the 'main.py' file.")