# For functions related to loading and preprocessing your image and video data.
import os

def get_image_paths(yoga_dir):
    """
    Function to get paths of all yoga pose images per yoga pose sub-directory.
    Argument: 
        yoga_dir : directory where yoga directories are stored
    """
    yoga_image_paths = {}
    
    # Iterate through each yoga pose directory
    for yoga_pose in os.listdir(yoga_dir):
        pose_dir = os.path.join(yoga_dir, yoga_pose)
        yoga_images = os.listdir(pose_dir)
        yoga_images = [os.path.join(pose_dir, image) for image in yoga_images]
        
        # Store in dictionary with pose name as key
        yoga_image_paths[yoga_pose] = yoga_images
    return yoga_image_paths

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