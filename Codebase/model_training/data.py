# Importing the enum module to create enumerations.
import enum

# Importing List and NamedTuple types from the typing module.
from typing import List, NamedTuple

# Importing the NumPy library and renaming it as np.
import numpy as np

# Defining an enumeration class called BodyPart.
class BodyPart(enum.Enum):
    # Enum members representing different body parts with assigned integer values.
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

# Defining a NamedTuple class called Point to represent 2D coordinates.
class Point(NamedTuple):
    x: float
    y: float

# Defining a NamedTuple class called Rectangle to represent a rectangle with two corners.
class Rectangle(NamedTuple):
    start_point: Point
    end_point: Point

# Defining a NamedTuple class called KeyPoint to represent a body keypoint with associated body part, coordinate, and score.
class KeyPoint(NamedTuple):
    body_part: BodyPart
    coordinate: Point
    score: float

# Defining a NamedTuple class called Person to represent a person with keypoints, bounding box, score, and an optional identifier.
class Person(NamedTuple):
    keypoints: List[KeyPoint]
    bounding_box: Rectangle
    score: float
    id: int = None

# Defining a function called person_from_keypoints_with_scores that takes keypoints_with_scores array, image height, image width,
# and an optional keypoint_score_threshold, and returns a Person object.
def person_from_keypoints_with_scores(
    keypoints_with_scores: np.ndarray,
    image_height: float,
    image_width: float,
    keypoint_score_threshold: float = 0.1) -> Person:
    
    # Extracting the x-coordinates of keypoints from the input array.
    kpts_x = keypoints_with_scores[:, 1]
    
    # Extracting the y-coordinates of keypoints from the input array.
    kpts_y = keypoints_with_scores[:, 0]
    
    # Extracting the scores of keypoints from the input array.
    scores = keypoints_with_scores[:, 2]
    
    # Initializing an empty list to store the extracted keypoints.
    keypoints = []
    
    # Looping over each keypoint score.
    for i in range(scores.shape[0]):
        # Creating a KeyPoint object with the corresponding body part, coordinate, and score, and adding it to the keypoints list.
        keypoints.append(
            KeyPoint(
                BodyPart(i),
                Point(int(kpts_x[i] * image_width), int(kpts_y[i] * image_height)),
                scores[i]))
    
    # Calculating the start point of the bounding box based on the minimum x and y coordinates among keypoints.
    start_point = Point(
        int(np.amin(kpts_x) * image_width), int(np.amin(kpts_y) * image_height))
    
    # Calculating the end point of the bounding box based on the maximum x and y coordinates among keypoints.
    end_point = Point(
        int(np.amax(kpts_x) * image_width), int(np.amax(kpts_y) * image_height))
    
    # Creating a Rectangle object representing the bounding box of the person.
    bounding_box = Rectangle(start_point, end_point)
    
    # Filtering the scores to only keep those above the keypoint_score_threshold.
    scores_above_threshold = list(
        filter(lambda x: x > keypoint_score_threshold, scores))
    
    # Calculating the average score of the keypoints above the threshold.
    person_score = np.average(scores_above_threshold)

    # Returning a Person object with the extracted keypoints, bounding box, and calculated person score.
    return Person(keypoints, bounding_box, person_score)

# Defining a NamedTuple class called Category to represent a category with a label and an associated score.
class Category(NamedTuple):
    label: str
    score: float