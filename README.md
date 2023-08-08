# Human Pose Estimation with MoveNet

This project implements a real-time human pose estimation pipeline using the MoveNet model from TensorFlow. It detects poses in images and videos, outputs keypoint locations, and classifies yoga poses.

## Contents

- [About](#about)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation) 
- [Methodology](#methodology)
  - [Data](#data)
  - [Preprocessing](#preprocessing)
  - [Model](#model)
  - [Pipeline](#pipeline)
  - [Evaluation](#evaluation)
  - [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About 

This project implements a pose estimation pipeline using MoveNet, a lightweight model from TensorFlow, for real-time detection and classification of yoga poses and exercises. The goals of this project were to:

- Accurately detect 2D pose keypoints in images and videos 
- Classify common yoga poses like tree, warrior, etc.
- Optimize model for real-time inference on edge devices
- Build an end-to-end pipeline from data to deployment

### Built With

- [TensorFlow](https://www.tensorflow.org/)
- [MoveNet](https://www.tensorflow.org/lite/models/pose_estimation/overview)
- [TensorFlow Lite](https://www.tensorflow.org/lite) 
- [Python]
- [OpenCV]
- [React]

## Getting Started

### Prerequisites

Python 3.8 or higher with pip installed.

### Installation
 
1. Clone the repo
   ```sh
   git clone https://github.com/JammalAdeyemi/Exercise-Pose-Detection.git
   ```
3. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```

## Methodology

### Data
- 1000 yoga pose images across 6 classes
- Exercise videos & Images - 6 classes 

### Preprocessing
- Load and merge CSVs
- Normalize keypoint coordinates 
- Train/val/test split

### Model
- MoveNet Thunder pretrained model
- Optimized for edge devices
- Input: Image/Video, Output: 17 Body Keypoints 

### Pipeline
- Load data
- Detect poses with MoveNet 
- Postprocess keypoints
- Construct dataset
- Train classifier 

### Evaluation
- Accuracy: 97% on pose classification

### Deployment
- Exported to TensorFlow Lite
- Integrated into React Web App

## Results (In Progress)

- Achieved X% accuracy in classifying Y yoga poses
- Inference time of Z ms per image using MoveNet on a laptop

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Your Name - [@Omotoyosi_Ade](https://twitter.com/Omotoyosi_Ade) - oabass7@gmail.com

Project Link: [https://github.com/JammalAdeyemi/Exercise-Pose-Detection](https://github.com/JammalAdeyemi/Exercise-Pose-Detection)

## Acknowledgements

- [TensorFlow MoveNet Model](https://www.tensorflow.org/lite/models/pose_estimation/overview)
- [React Webcam](https://github.com/mozmorris/react-webcam)