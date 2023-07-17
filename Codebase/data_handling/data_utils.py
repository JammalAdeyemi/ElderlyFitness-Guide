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

# Get video paths for a folder
def get_video_paths(folder):
  # List all MP4 videos in the folder
  videos = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
  return videos