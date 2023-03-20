#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------
# in term of opencv, I choose the opencv_python version that match python version
import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import os
import torch

import warnings

warnings.filterwarnings("ignore")

from monitor import monitor
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer

#################################
# Start camera
#################################

cam_idx = 0

# adjust these for your camera to get the best accuracy
call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

cam_cap = cv2.VideoCapture(cam_idx)
cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# calibrate camera
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
if path.exists("calib_cam%d.pkl" % (cam_idx)):
    cam_calib = pickle.load(open("calib_cam%d.pkl" % (cam_idx), "rb"))
else:
    print(
        "Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
    print("Press s to save frame, c to continue, q to quit")
    cam_calibrate(cam_idx, cam_cap, cam_calib)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sys.path.append("./GazeTR_main")
from model import Model

gaze_network = Model().to(device)

gazeTR_weights = torch.load("/home/hxy/few_shot_gaze/gazeTR_demo/GazeTR-H-ETH.pt")
if torch.cuda.device_count() == 1:
    if next(iter(gazeTR_weights.keys())):
        gazeTR_weights = dict([(k, v) for k, v in gazeTR_weights.items()])
gaze_network.load_state_dict(gazeTR_weights)

# Initialize monitor and frame processor
mon = monitor()
frame_processor = frame_processer(cam_calib)

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
subject = 'a'
data = frame_processor.process(subject, cam_cap, mon, device, gaze_network, show=True)
