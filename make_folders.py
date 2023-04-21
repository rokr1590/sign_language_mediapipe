import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'thumbsup'])

# Ten videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 30
cap = cv2.VideoCapture(0)
start_folder = 0
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
