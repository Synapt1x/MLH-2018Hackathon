# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

face_detector.py: this contains the code for using a Haar Cascade face detection
classifier.
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


class FaceDetector:

    def __init__(self, config=None, curdir=None, type='face'):

        if config is not None:
            self.config = config
        self.detector = cv2.CascadeClassifier()
        if type == 'face':
            self.detector.load(os.path.join(curdir, 'haarcascade-features/haarcascade.xml'))
        elif type == 'body':
            self.detector.load(os.path.join(curdir, 'haarcascade-features/haarcascadebody.xml'))
        elif type == 'upper':
            self.detector.load(os.path.join(curdir, 'haarcascade-features/haarcascadeupper.xml'))
        elif type == 'profile':
            self.detector.load(os.path.join(curdir, 'haarcascade-features/haarcascadeprofile.xml'))

    def process_frame(self, frame, target_size=(640, 1140)):

        orig_frame = frame.copy()

        # rescale to gray
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # reshape to correct size
        if frame.shape[:2] != target_size:
            orig_frame = orig_frame[40: 680, 70: 1210]
            frame = frame[40: 680, 70: 1210]

        boxes = self.detector.detectMultiScale(frame)

        out_frame = orig_frame
        for (x, y, w, h) in boxes:
            if y > 540:
                continue
            out_frame = cv2.rectangle(orig_frame, (x, y), (x + w, y + h),
                                  color=(255, 0, 0), thickness=2)

        return out_frame, boxes


if __name__ == '__main__':
    print("Please run the 'main.py' file.")
