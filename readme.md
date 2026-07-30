Emotion Detector
Trinity Hu
My program uses a computer vision system and video camera to analyze facial expressions in real time. My trained AI model classifies emotions into four categories: happy, sad, neutral, and shocked.
Images:
<img width="1612" height="1412" alt="Screenshot 2026-07-29 175137" src="https://github.com/user-attachments/assets/6d95e45f-31ed-4417-8d22-c2671d179778" />
<img width="1621" height="1420" alt="Screenshot 2026-07-29 175023" src="https://github.com/user-attachments/assets/193ec359-9101-4f5e-8dc5-212e13fe047d" />
<img width="1636" height="1426" alt="Screenshot 2026-07-29 174624" src="https://github.com/user-attachments/assets/ce8b73b6-6def-47e6-894d-6dfde1fc5f3a" />
<img width="1615" height="1403" alt="Screenshot 2026-07-29 174323" src="https://github.com/user-attachments/assets/f71929ad-8129-4501-9e3e-9088dc5adc0d" />

The Algorithm

This project uses a deep learning image classification algorithm to recognize human emotions from camera images. The model is based on ResNet18.

How it was trained
  1. A dataset containing images of different emotions is collected (in this case I used my own pictures)
  2. Images are separated into training and validation folders
  3. The ResNet18 model learns patterns from the training images
  4. The model is tested using validation images
  5. The trained model is exported as an ONNX file
  6. The ONNX model is loaded onto the Jetson Nano for real time detection.

How the code works
  1. The camera input is handled using OpenCV
  2. The image is passed into the ResNet18 model
  3. TensorRT optimizes the model so it can run efficiently on the Jetson Nano GPU
  4. The model returns a class prediction
  5. A labels file (labels.txt) converts the class number into an emotion name

Dependencies: Python 3, NumPy, Pillow, OpenCV, PyTorch, Torchvision, Jetson Inference, TensorRT, CUDA, cuDNN

Running this project
  1. Open terminal on Jetson Nano
  2. Install the required libraries: Python 3, pip, NumPy, Pillow, OpenCV, Jetson Inference, TensorRT, CUDA, cuDNN, PyTorch (for training/exporting), Torchvision (for training/exporting)
  3. Go to the ONNX ResNet18 model location: /home/nvidia/jetson-inference/python/training/classification/models/emotions/
  4. Connect a camera directly to the Jetson Nano
  5. Now run the program on Rustdesk and a camera window should open
  6. An emotion will show real time on the top left corner of that window

Video explanation link:

