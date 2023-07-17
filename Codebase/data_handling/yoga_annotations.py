import cv2
import os
import mediapipe as mp
import pandas as pd
from data_utils import load_images

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

folder_path = "../../Data_dissertation/Yoga_Datasets"
folders = [os.path.join(folder_path, f) for f in ['Anjaneyasana', 'Baddha_Konasana', 'Marjaryasana', 'Savasana',
                                                  'Supta_matsyendrasana', 'tadasana', 'urdhva_hastasana', 'Vrksasana']]

# Get the list of all the pose landmarks
keypoints = [l.name for l in mp_pose.PoseLandmark]
# Create empty lists to store the pose landmarks and the pose nameß
landmarks = []
poses = []

for folder in folders:
    pose_name = os.path.basename(folder)
    for image_path in load_images(folder):
        # get image path
        image = cv2.imread(image_path)
        # convert to RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # perform pose detection
        results = pose.process(imageRGB)
        row = [pose_name]

        # Check if any pose was detected
        if results.pose_landmarks:
            # Append x, y, z coords
            for landmark in results.pose_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
            landmarks.append(row)
            poses.append(pose_name)

# Create separate columns for each coordinate of each pose landmark
column_names = ["pose_name"]
# Iterate over each pose landmark
for kp in keypoints:
  for coord in ['x', 'y', 'z']:
    column_names.append(f"{coord}_{kp}")

# Create DataFrame
df = pd.DataFrame(landmarks, columns=column_names)
df.to_csv('../../Data/yoga_pose.csv', index=False)

