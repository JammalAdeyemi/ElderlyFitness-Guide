# FitnessPal for Tiago: An Elderly Exercise and Yoga Buddy

## Overview
FitnessPal for Tiago is an innovative project aimed at providing tailored exercise and yoga support for elderly individuals, leveraging state-of-the-art pose estimation technology. Our system encourages physical well-being, enhances engagement and adherence to exercise and yoga routines, and improves form and balance.

## Contents

- [About](#about)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation) 
- [How It Works](#how-it-works)
- [Methodology](#methodology)
  - [Data](#data)
  - [Preprocessing](#preprocessing)
  - [Model](#model)
  - [Pipeline](#pipeline)
  - [Evaluation](#evaluation)
  - [Deployment](#deployment)
- [Key Features](#key-features)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About 

This project implements a pose estimation pipeline using MoveNet, a lightweight model from TensorFlow, for real-time detection and classification of yoga poses and exercises. The goals of this project were to:

- Accurately detect 2D pose keypoints in images and videos 
- Classify yoga and exercise pose that are easy for the elders.
- Build a model that can accurately classify yoga and exercises poses.
- Build a web-app using react, tensorflow.js, opencv.js and integrate the developed model to the web-app.
- Perform user testing to evaluate the sytem's performance

### Built With

- [TensorFlow](https://www.tensorflow.org/)
- [MoveNet](https://www.tensorflow.org/lite/models/pose_estimation/overview)
- [TensorFlow Lite](https://www.tensorflow.org/lite) 
- Python
- OpenCV
- React

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

## How It Works
1. Pose Detection: We gather exercise and yoga data from various sources and apply Movenet for keypoint extraction.
2. Data Preprocessing: We preprocess the extracted keypoints to create a robust pose detection model.
3. Deep Learning Models: We build and evaluate different pose classification models (CNN, DNN, MLP) to choose the best performer.
4. Web Application: The selected model is integrated into our user-friendly web app, offering real-time feedback.
5. Deployment: Our system is deployed on the Tiago robot for real-time assistance to elderly users.

## Methodology

### Data
- 1000 yoga pose images across 6 classes from [Kaggle](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset?select=Adho+Mukha+Svanasana)
- Exercise videos & Images - 6 classes from youtube videos and google images

### Preprocessing
- Load and merge CSVs
- Normalize keypoint coordinates 
- Train/val/test split

### Model
- MoveNet Thunder pretrained model
- CNN
- DNN
- MLP

### Pipeline
- Load data
- Detect poses with MoveNet 
- Preprocess keypoints
- Construct dataset
- Train classifier 

### Evaluation
- Accuracy: 98% on pose classification

### Deployment
- Exported to TensorFlow Lite
- Integrated into React Web App
- Deployed the web app on Tiago robot

## Key Features
- Pose Detection: Utilizing advanced pose estimation models, we offer real-time feedback and posture correction.
- Repetition Counting: Our system reliably tracks exercise and yoga repetitions, ensuring effectiveness.
- User-Friendly Interface: The application is designed for easy interaction, making it accessible to elderly users.
- Deployment on Tiago: FitnessPal is integrated into the Tiago robot, allowing real-time interaction and assistance.

## Future Work
- Personalized Routines: Implementing personalized exercise routines based on user profiles.
- Enhanced Interactivity: Adding voice commands and conversational AI for more natural interaction.
- Continuous Improvement: Regular updates and enhancements to improve user satisfaction.

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