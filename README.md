🔍 Gender and Age Detection Using OpenCV and Deep Learning
This mini project detects a person's gender and age group from an image using OpenCV and pre-trained deep learning models.

🚀 Features
Detects gender: Male or Female

Predicts age group from predefined ranges

Uses pre-trained Caffe models on the Adience dataset

Lightweight and easy to run with Python and OpenCV



📁 Files
age_deploy.prototxt / age_net.caffemodel: Age prediction model

gender_deploy.prototxt / gender_net.caffemodel: Gender prediction model

main.py or age_gender_detection.py:  main Python script

your_image.jpg: Example image (can be replaced with any face image)



🔧 Requirements
Python 3.x

OpenCV

NumPy


🧠 Pre-trained Models
Models are downloaded automatically when you run the script.

Gender model trained on Adience dataset

Age model trained on 8 age ranges:
(0-2), (4-6), (8-12), (15-20),
(25-32), (38-43), (48-53), (60-100)



💡 Future Improvements
Add face detection (e.g., Haar cascade or DNN face detector)

Extend to real-time webcam prediction

Build a simple GUI using Tkinter or Streamlit



