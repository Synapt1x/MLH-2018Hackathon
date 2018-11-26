# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at University of Manitoba.

cnn.py: this contains the code for a neural network model.
"""

##############################################################################
__author__ = ["Chris Cadonic", "Cassandra Aldaba"]
__credits__ = ["Chris Cadonic", "Cassandra Aldaba"]
__version__ = "0.1"
##############################################################################


import os

import pickle
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, \
    ZeroPadding2D, Flatten, Dense, Dropout


class NeuralNet:

    def __init__(self, config=None):

        if config is not None:
            self.config = config

    #TODO: finish rest


if __name__ == '__main__':
    print("Please run the 'main.py' file.")