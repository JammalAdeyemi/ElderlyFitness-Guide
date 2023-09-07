import cv2
import os


video_path = cv2.VideoCapture('/Users/oabas/Documents/GitHub/Exercise-Pose-Detection/Data_dissertation/Exercise_videos/Sit_to_stand/videoplayback.mp4')
# Variable to keep track of the current frame number
currentframe = 0 
skip_interval = 0.5

# Checking if the directory to store extracted images exists or not
try:
    if not os.path.exists('../../../Data_dissertation/Exercise_images/Sit_to_stand-7'):
        os.makedirs('../../../Data_dissertation/Exercise_images/Sit_to_stand-7')
# Printing an error message if there is an exception while creating the directory
except OSError:
    print('Error: Creating directory of data')

fps = video_path.get(cv2.CAP_PROP_FPS)  # Getting the frames per second (FPS) of the video
skip_interval_frame = int(skip_interval * fps)

while True:
    ret, frame = video_path.read()
    if ret:
        name = '../../../Data_dissertation/Exercise_images/Sit_to_stand-7/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        currentframe += 1
        
        for _ in range(skip_interval_frame):
            video_path.read() # Skipping frames
    else:
        break

video_path.release()
cv2.destroyAllWindows()
