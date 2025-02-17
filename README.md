# Yoga-Pose-Estimation-and-Feedback-Generation

## Overview
This project detects yoga poses in real-time and provides corrective feedback to improve alignment and accuracy. It analyzes body postures, calculates angles, and offers visual and auditory cues for better practice.

## Features
- **Real-time Pose Detection** using MediaPipe.
- **Angle Calculation** for evaluating pose accuracy.
- **Feedback Generation** with on-screen and audio guidance.
- **User-friendly Interface** for real-time corrections.

## Technologies Used
Python, MediaPipe, OpenCV, NumPy, Pandas, Matplotlib, Pickle, pyttsx3

## Data & Model
- Collected **55 videos** covering five yoga poses.
- Extracted landmarks and stored in **CSV format**.
- Trained **Random Forest, Ridge Classifier, Logistic Regression, Gradient Boosting** machine learning models using the extracted landmarks.
- Achieved **95% accuracy** in pose classification.

## Feedback Mechanism
- Compared **predicted vs. actual pose angles**.
- Provided real-time **visual and voice feedback** for corrections.

## Future Enhancements
- Expanding pose database and development of a mobile or web application
