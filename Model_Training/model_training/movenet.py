# Importing necessary libraries and modules.
import os
from typing import Dict, List

import cv2
from data import BodyPart, Person, person_from_keypoints_with_scores
import numpy as np

try:
    # Trying to import the tflite_runtime.interpreter module, if not available, import tensorflow's tflite.Interpreter.
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Class representing the MoveNet object.
class Movenet(object):

    # Constants used in the class.
    _MIN_CROP_KEYPOINT_SCORE = 0.2
    _TORSO_EXPANSION_RATIO = 1.9
    _BODY_EXPANSION_RATIO = 1.2

    # Constructor method for initializing the MoveNet object.
    def __init__(self, model_name: str) -> None:
        _, ext = os.path.splitext(model_name)
        if not ext:
            model_name += '.tflite'

        # Initialize the model with the provided tflite file.
        interpreter = Interpreter(model_path=model_name, num_threads=4)
        interpreter.allocate_tensors()

        self._input_index = interpreter.get_input_details()[0]['index']
        self._output_index = interpreter.get_output_details()[0]['index']

        self._input_height = interpreter.get_input_details()[0]['shape'][1]
        self._input_width = interpreter.get_input_details()[0]['shape'][2]

        self._interpreter = interpreter
        self._crop_region = None

    # Method to initialize the crop region based on the input image dimensions.
    def init_crop_region(self, image_height: int, image_width: int) -> Dict[str, float]:
        if image_width > image_height:
            x_min = 0.0
            box_width = 1.0
            # Pad the vertical dimension to become a square image.
            y_min = (image_height / 2 - image_width / 2) / image_height
            box_height = image_width / image_height
        else:
            y_min = 0.0
            box_height = 1.0
            # Pad the horizontal dimension to become a square image.
            x_min = (image_width / 2 - image_height / 2) / image_width
            box_width = image_height / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    # Method to check if the torso keypoints are visible.
    def _torso_visible(self, keypoints: np.ndarray) -> bool:
        left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
        right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
        left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
        right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

        left_hip_visible = left_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        right_hip_visible = right_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        left_shoulder_visible = left_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        right_shoulder_visible = right_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE

        return ((left_hip_visible or right_hip_visible) and
                (left_shoulder_visible or right_shoulder_visible))

    # Method to determine the range of torso and body keypoints from the center point.
    def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                        target_keypoints: Dict[str, float],
                                        center_y: float,
                                        center_x: float) -> List[float]:
        torso_joints = [
            BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_HIP,
            BodyPart.RIGHT_HIP
        ]
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for idx in range(len(BodyPart)):
            if keypoints[BodyPart(idx).value, 2] < Movenet._MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[BodyPart(idx)][0])
            dist_x = abs(center_x - target_keypoints[BodyPart(idx)][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange
        ]

    # Method to determine the crop region based on keypoints and image dimensions.
    def _determine_crop_region(self, keypoints: np.ndarray, image_height: int,
                               image_width: int) -> Dict[str, float]:
        target_keypoints = {}
        for idx in range(len(BodyPart)):
            target_keypoints[BodyPart(idx)] = [
                keypoints[idx, 0] * image_height, keypoints[idx, 1] * image_width
            ]

        if self._torso_visible(keypoints):
            center_y = (target_keypoints[BodyPart.LEFT_HIP][0] +
                        target_keypoints[BodyPart.RIGHT_HIP][0]) / 2
            center_x = (target_keypoints[BodyPart.LEFT_HIP][1] +
                        target_keypoints[BodyPart.RIGHT_HIP][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange,
            max_body_xrange) = self._determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * Movenet._TORSO_EXPANSION_RATIO,
                max_torso_yrange * Movenet._TORSO_EXPANSION_RATIO,
                max_body_yrange * Movenet._BODY_EXPANSION_RATIO,
                max_body_xrange * Movenet._BODY_EXPANSION_RATIO
            ])

            distances_to_border = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(distances_to_border)])

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
            return {
                'y_min':
                    crop_corner[0] / image_height,
                'x_min':
                    crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                        crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                    crop_corner[1] / image_width
            }
        else:
            return self.init_crop_region(image_height, image_width)

    # Method to crop and resize the image based on the crop region.
    def _crop_and_resize(
            self, image: np.ndarray, crop_region: Dict[str, float],
            crop_size: (int, int)) -> np.ndarray:
        y_min, x_min, y_max, x_max = [
            crop_region['y_min'], crop_region['x_min'], crop_region['y_max'],
            crop_region['x_max']
        ]

        crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
        crop_bottom = int(image.shape[0] if y_max >= 1 else y_max * image.shape[0])
        crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
        crop_right = int(image.shape[1] if x_max >= 1 else x_max * image.shape[1])

        padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
        padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
        padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
        padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)

        # Crop and resize image
        output_image = image[crop_top:crop_bottom, crop_left:crop_right]
        output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                        padding_left, padding_right,
                                        cv2.BORDER_CONSTANT)
        output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

        return output_image

    # Method to run the MoveNet detector on the input image with the specified crop region.
    def _run_detector(
            self, image: np.ndarray, crop_region: Dict[str, float],
            crop_size: (int, int)) -> np.ndarray:
        input_image = self._crop_and_resize(image, crop_region, crop_size=crop_size)
        input_image = input_image.astype(dtype=np.uint8)

        self._interpreter.set_tensor(self._input_index,
                                    np.expand_dims(input_image, axis=0))
        self._interpreter.invoke()

        keypoints_with_scores = self._interpreter.get_tensor(self._output_index)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        for idx in range(len(BodyPart)):
            keypoints_with_scores[idx, 0] = crop_region[
                'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
            keypoints_with_scores[idx, 1] = crop_region[
                'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]

        return keypoints_with_scores

    # Method to detect a person in the input image using MoveNet.
    def detect(self,
            input_image: np.ndarray,
            reset_crop_region: bool = False) -> Person:
        image_height, image_width, _ = input_image.shape
        if (self._crop_region is None) or reset_crop_region:
            self._crop_region = self.init_crop_region(image_height, image_width)

        keypoints_with_scores = self._run_detector(
            input_image,
            self._crop_region,
            crop_size=(self._input_height, self._input_width))
        self._crop_region = self._determine_crop_region(keypoints_with_scores,
                                                        image_height, image_width)

        return person_from_keypoints_with_scores(keypoints_with_scores, image_height,
                                                image_width)
