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

    #TODO: finish rest


if __name__ == '__main__':
    print("Please run the 'main.py' file.")