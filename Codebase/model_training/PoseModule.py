# For creating and training the pose estimation model on data images.
import cv2
import mediapipe as mp
import time
import math

class pose_detector():
    # Constructor
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
    
    # Find the pose.
    def find_pose(self, img, draw=True):
        # Convert the BGR image to RGB.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Perform Pose detection.
        self.results = self.pose.process(imgRGB)
        # Check for landmarks.
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img
    
    # Find the position of the landmarks.
    def find_position(self, img, draw=True):
        self.landmarks = []
        if self.results.pose_landmarks:
            # Print the landmarks.
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Retrieve the width and height of the image.
                height, width, _ = img.shape
                # Retrieve the x, y and z co-ordinates of the landmark.
                cx, cy, cz = int(lm.x * width), int(lm.y * height), int(lm.z * width)
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy, cz), 5, (255, 0, 0), cv2.FILLED)
            return self.landmarks
        
    # Given any 3 points, find the angle between them.
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks.
        x1, y1 = self.landmarks[p1][1:]
        x2, y2 = self.landmarks[p2][1:]
        x3, y3 = self.landmarks[p3][1:]
        
        # Calculate the angle.
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - 
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        print(f"Angle: {angle}")
              
        # Draw the angle.
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            
            # Put the angle on the screen.
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2
                        )
        return angle
        

def main():
    cap = cv2.VideoCapture('../../Data/1.mp4')
    detector = pose_detector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lmList = detector.find_position(img,draw=True)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
            
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()