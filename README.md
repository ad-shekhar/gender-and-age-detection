🎯 Gender & Age Detection using Deep Learning

A lightweight deep-learning powered system that predicts a person’s gender and age range from an image or live webcam feed using OpenCV and pre-trained models.

📌 Table of Contents

Objective

Project Overview

Dataset

Requirements

Project Structure

Usage

Examples

Demo Video

Features

License

🎯 Objective

The purpose of this project is to build a Gender and Age Detector capable of identifying a person's gender (Male/Female) and age group from:

A single face image

A real-time webcam feed

Predicted age groups include:

0–2, 4–6, 8–12, 15–20, 25–32, 38–43, 48–53, 60–100 years

These ranges reflect the classification categories from the pre-trained model.

📘 Project Overview

This project uses:

OpenCV DNN module for deep-learning inference

Caffe models for age & gender prediction

TensorFlow model for face detection

Softmax classifier for final predictions

It was designed to be simple, accurate, and fast enough to run on CPU-only systems.

📚 Dataset

The model is based on the Adience Benchmark Dataset, a widely used dataset for age and gender classification research.

🔗 Dataset link:
https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification

The dataset contains:

26,000+ photos

2,284 subjects

Multiple lighting, pose, and occlusion variations

8 distinct age categories

📦 Requirements

Install dependencies:

pip install opencv-python
pip install argparse

📂 Project Structure
├── detect.py # Main detection script
├── age_deploy.prototxt # Age model structure
├── age_net.caffemodel # Age model weights
├── gender_deploy.prototxt # Gender model structure
├── gender_net.caffemodel # Gender model weights
├── opencv_face_detector.pb # Face detection model
├── opencv_face_detector.pbtxt # Face detection model config
├── Example/ # Example output images
├── \*.jpg # Sample images for testing

▶️ Usage
1️⃣ Detect gender & age from an image

Ensure the image is in the same folder.

python detect.py --image <image_name.jpg>

Example:

python detect.py --image girl1.jpg

2️⃣ Detect gender & age using webcam
python detect.py

Press Ctrl + C to stop webcam mode.

🖼️ Examples

> python detect.py --image girl1.jpg
> Gender: Female
> Age: 25–32 years

<img src="Example/Detecting age and gender girl1.png">
> python detect.py --image man2.jpg
Gender: Male
Age: 25–32 years

<img src="Example/Detecting age and gender man2.png">

Note: Example images are used only for educational purposes. If any copyright concerns arise, they can be removed.

💡 Features

✔ Real-time gender & age prediction
✔ Works with images and webcam
✔ No GPU required
✔ Pre-trained deep-learning models
✔ Accurate predictions across age groups
✔ Minimal dependencies & easy to run

📜 License

This project is open-source and available under the MIT License.
