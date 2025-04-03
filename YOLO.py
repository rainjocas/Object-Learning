"""
@software{yolov5,
  title = {YOLOv5 by Ultralytics},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}
"""
import os
import pandas as pd
import torch
import numpy as np
import torchvision.io as io
import cv2
from os import listdir
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def runYOLOSmBatch():
  """ Runs YOLO on a small batch of images, saves the newly bounded images"""
  # Batch of images
  images = ["Media/Crows.jpg", "Media/Dog Park.jpg",
            "Media/Petting Zoo.jpg", "Media/Street.jpg",
            "Media/Cars Movie.png"]

  # Inference
  results = model(images)

  # Save new images with bounding boxes and classes
  results.save()

  # Results
  results.print()

#Load Busses dataset
IMAGE_ROOT = 'archive/images'
df = pd.read_csv('archive/df.csv')
print("First 5 records:", df.head())

#grab filenames
fileNames = os.listdir('archive/images/images')

#convert filenames to file paths
imgs = []

for fileName in fileNames:
    filePath = 'archive/images/images/' + fileName
    imgs.append(filePath)
    #print(img)

print("Images:", imgs[0:5])

#testing on smaller set because it keeps crashing
imgs = imgs[0:5]

# Inference
results = model(imgs)
# results = model(imgs).to(device)

print("ELEPHANT")

# Results
results.print()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

print(results.pandas().xyxy[0])

def getConfidences():
  """ Returns a list of average confidence for all images """
  confidences = []
  for i in range(len(imgs)):
      #get confidences for an image
      confidence = results.pandas().xyxy[i].confidence
      #average all confidences in an image & add to list
      avgConfidence = sum(confidence)/ len(confidence)
      confidences.append(avgConfidence)
  return confidences

def getIoUs():
  """ Returns a list of average confidence for all images """
  IoUs = []
  # for i in range(len(imgs)):
    #get IoUs for an image


      #IoUs.append(avgConfidence)
  return IoUs

def getDetectionAcc(confidences, IoUs):
  """ Calculates and returns the detection accuracy for cofidence levels and IoU """
  confThreshold = 0.8
  IoUThreshold = 0.8

  #get number of images that pass the confidence threshold
  totPassedConf = 0
  for conf in confidences:
      if conf >= confThreshold:
          totPassedConf += 1

  #detection accuracy for confidence levels
  confAcc = totPassedConf/len(confidences)

  #get number of images that pass the IoU threshold
  totPassedIoU = 0
  for IoU in IoUs:
      if IoU >= IoUThreshold:
          totPassedIoU += 1

  #detection accuracy for IoU
  IoUAcc = totPassedIoU/len(IoUs)

  return (confAcc, IoUAcc)

confidences = getConfidences()
IoUs = getIoUs()

confAcc, IoUAcc = getDetectionAcc(confidences, IoUs)


"""
You can establish a threshold value for confidence level and another for IoU that
considers the detection successful or failed. Then compute the detection accuracy
based on that. In order to compute the IoU between two bounding boxes, write a
function compute_IoU that does that.
"""