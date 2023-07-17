# For functions related to loading and preprocessing your image and video data.
import os

def load_images(folder):
  image_paths = []
   # Iterate through each image file in the folder
  for filename in os.listdir(folder):
      # Checking if the file ends with .jpg or .png
    if filename.endswith('.jpg') or filename.endswith('.png'):
      image_path = os.path.join(folder, filename)  
      image_paths.append(image_path)
  return image_paths

def get_video_paths(exercise_dir):
    """
    Function to get paths of all exercise videos per exercise sub-directory.
    Argument: 
        exercise_dir : directory where exercise directories are stored
    """
    exercise_video_paths = {}
    
    # Iterate through each exercise directory
    for exercise in os.listdir(exercise_dir):
        exercise_subdir = os.path.join(exercise_dir, exercise)
        exercise_videos = os.listdir(exercise_subdir)
        exercise_videos = [os.path.join(exercise_subdir, video) for video in exercise_videos]
        
        # Store in dictionary with exercise name as key
        exercise_video_paths[exercise] = exercise_videos
    return exercise_video_paths