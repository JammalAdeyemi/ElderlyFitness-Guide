import cv2  
import os  

# Initializing the video capture object with the path of the video file
video_path = cv2.VideoCapture('../../../Data_dissertation/Exercise_videos/Sit_to_stand/videoplayback.mp4')  
currentframe = 0  # Variable to keep track of the current frame number
start_time = 20 
skip_interval = 0.5
# end_time = 50  

# Checking if the directory to store extracted images exists or not
try:
    if not os.path.exists('../../../Data_dissertation/Exercise_images/Sit_to_stand-7'):  
        os.makedirs('../../../Data_dissertation/Exercise_images/Sit_to_stand-7')  # Creating the directory if it doesn't exist
# Printing an error message if there is an exception while creating the directory
except OSError:
    print('Error: Creating directory of data')  

fps = video_path.get(cv2.CAP_PROP_FPS)  # Getting the frames per second (FPS) of the video
start_frame = int(start_time * fps)  # Calculating the starting frame based on the FPS and start time
skip_interval_frame = int(skip_interval * fps)
#end_frame = int(end_time * fps)  # Calculating the ending frame based on the FPS and end time

# Setting the video capture object to start from the desired frame
video_path.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  

while True:
    ret, frame = video_path.read()  # Reading the next frame from the video
    if ret: #and currentframe <= end_frame:  # Checking if the frame was read successfully and if the current frame is within the desired range
        name = '../../../Data_dissertation/Exercise_images/Sit_to_stand-7/frame' + str(currentframe) + '.jpg'  
        print('Creating...' + name) 
        cv2.imwrite(name, frame)  # Saving the frame as an image file
        currentframe += 2  
        
        for _ in range(skip_interval_frame):
            video_path.read() # Skipping frames
    else:
        break  

video_path.release()  
cv2.destroyAllWindows()  
