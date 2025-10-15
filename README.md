# Emotion Detection from Facial Expressions

This project implements a real-time pipeline for detecting human emotions from facial expressions using a deep learning approach. The system integrates a YOLOv11 model for face detection and an improved GoogleNet architecture for emotion classification, achieving a test accuracy of 69.63% on the challenging FER2013 dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Results](#results)
- [Usage](#usage)

## Project Overview
Facial expressions are among the most powerful indicators of human emotion, but interpreting them reliably, especially by a machine, remains a complex task. This system tackles the challenge of mapping predefined emotion categories to subtle facial features. The project involved engineering a robust pipeline that integrates an improved GoogleNet architecture for emotion classification with the YOLOv11 face localization object detection model in real time, using the FER2013 dataset. **Temporal smoothing** was a key innovation, where predictions are collected within a 30-frame buffer and the majority-vote prediction is returned as the output. This approach effectively reduces frame-wise prediction noise and transient errors, producing a stable representation of emotional states, as a single frame is often insufficient to detect an emotion due to the intricate nature of human expressions. Another important enhancement was the adoption of the FocalLoss function, which addressed extreme class imbalance and led to a significant jump in performance, from 54.61% test accuracy with Cross-Entropy Loss to 69.63% with FocalLoss.

## Dataset
The model was trained on the **FER2013 dataset**, which consists of 35,887 grayscale images of 48x48 pixels. The dataset is known for its difficulty as it contains "in-the-wild" images with variations in head poses, illumination, and occlusions.

The project covers six primary emotion categories: Angry, Fear, Happy, Sad, Surprise, and Neutral. The "Disgust" class was removed due to its high ambiguity and skewed representation in the dataset.

You can download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

## Results
The final model achieved a **test accuracy of 69.63%**, a significant improvement over the baseline CNN models. The Focal Loss function and careful data adjustments helped improve performance across the runs.

| Model                             | Epochs | Patience | Loss              | Accuracy |
|----------------------------------|-------|---------|-----------------|---------|
| GoogleNet Run 1                   | 30    | 5       | CrossEntropy Loss | 50.72%  |
| GoogleNet Run 2                   | 50    | 10      | CrossEntropy Loss | 54.61%  |
| GoogleNet Run 3                   | 50    | 10      | Focal Loss        | 67.89%  |
| GoogleNet Run 4 (Substitute Data) | 50    | 10      | Focal Loss        | 68.70%  |
| GoogleNet Run 5 (Removed Class)   | 50    | 10      | Focal Loss        | 69.63%  |

## Usage
To use the system, you can either **run the pre-trained models** or **retrain the models** as preferred:

1. **Run the pre-trained models**  
   - `best_googlenet_emotion_model.pth` (Run 3)  
   - `UPDATEDGOOGLENETFINAL.pth` (Run 5)  
   - Open and run `YOLOV11.ipynb`. This notebook will handle face detection and real-time emotion classification using your webcam or video file.

2. **Retrain the models (optional)**  
   - Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).  
   - Open `Run 3.ipynb` or `Run 5.ipynb`.  
   - Update dataset paths and run the notebook to train your own models. The notebooks will install the required packages automatically as needed.
