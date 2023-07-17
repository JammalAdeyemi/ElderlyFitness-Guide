import cv2
import os
import mediapipe as mp
import pandas as pd
from data_utils import get_video_paths

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

folder_path = "../../Data_dissertation/Exercise"
folders = [os.path.join(folder_path, f) for f in ['arm_raise', 'bicycle_crunch', 'bird_dog', 'curl', 'fly', 'leg_raise', 
                                                  'pushup', 'squat', 'overhead_press', 'superman']]

# Get the list of all the pose landmarks
keypoints = [l.name for l in mp_pose.PoseLandmark]
# Create empty lists to store the pose landmarks and the pose nameß
landmarks = []
exercise = []

for folder in folders:
    exercise_name = os.path.basename(folder)
    for video_path in get_video_paths(folder):
        # get video path
        video = cv2.VideoCapture(video_path)
        while True:
            success, frame = video.read()
            if not success:
                break  
             
            # process frame
            results = pose.process(frame)
            # create row
            row = [exercise_name]
            # append landmarks
            if results.pose_landmarks:
                for lmk in results.pose_landmarks.landmark:
                    row.append(lmk.x)
                    row.append(lmk.y)
                    row.append(lmk.z)
                # append row
                landmarks.append(row)
                # append exercise
                exercise.append(exercise_name)
                
        video.release()
        
# Create columns
column_names = ["exercise"]
# Iterate over each pose landmark
for kp in keypoints:
  for coord in ['x', 'y', 'z']:
    column_names.append(f"{coord}_{kp}")

# Create dataframe 
df = pd.DataFrame(landmarks, columns=column_names)

# Export CSV
df.to_csv('../../Data/exercises.csv', index=False)