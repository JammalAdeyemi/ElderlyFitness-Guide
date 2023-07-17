import cv2
import os
import mediapipe as mp
import pandas as pd
from data_utils import load_images

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)