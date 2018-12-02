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
#import lk
#import cnn

import opencvExample 
from tkinter import *
def main():
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

    # Get video path
    vid_dir = os.path.join(curdir, config['traindir'])
    vidPath = os.path.join(vid_dir, 'heartAttack_mpeg4.avi')

    # Open GUI
    opencvExample.App(Tk(),"ERVA (Emergency Room Video Assistant)", vidPath)  

if __name__ == '__main__':
    main()