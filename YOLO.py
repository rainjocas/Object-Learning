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

def getConfidences(imgs, results):
  """ Returns a list of average confidence for all images """
  confidences = []
  for i in range(len(imgs)):
      #get confidences for an image
      confidence = results.pandas().xyxy[i].confidence
      #average all confidences in an image & add to list
      avgConfidence = sum(confidence)/ len(confidence)
      confidences.append(avgConfidence)
  return confidences

def getCoords(img):
  """ Returns lists of coordinates for all bounding boxes in an image """
  xMins = img.xmin
  xMaxs = img.xmax
  yMins = img.ymin
  yMaxs = img.ymax

  xMins = xMins.to_list()
  xMaxs = xMaxs.to_list()
  yMins = yMins.to_list()
  yMaxs = yMaxs.to_list()

  predBBcoords = (xMins, xMaxs, yMins, yMaxs)
  
  return predBBcoords

def compute_IoU(df, imgID, predBBcoords, objInd, objCount):
  """ Computes the IoU of an object """
  #get true bounding coords for object
  trueXMin = df.loc[df['ImageID'] == imgID].XMin[objCount]
  trueXMax = df.loc[df['ImageID'] == imgID].XMax[objCount]
  trueYMin = df.loc[df['ImageID'] == imgID].YMin[objCount]
  trueYMax = df.loc[df['ImageID'] == imgID].YMax[objCount]
  
  #get predicted bounding coords for object
  predXMin = predBBcoords[0][objInd]
  predXMax = predBBcoords[1][objInd]
  predYMin = predBBcoords[2][objInd]
  predYMax = predBBcoords[3][objInd]

  #intersection
  interWidth = min(trueXMax, predXMax) - max(trueXMin, predXMin)
  interHeight = min(trueYMin, predYMin) - max(trueYMax, predYMax)
  interArea = interWidth * interHeight


  trueArea = abs(trueXMin - trueXMax) * abs(trueYMin - trueYMax)
  predArea = abs(predXMin - predXMax) * abs(predYMin - predYMax)
  unionArea = trueArea + predArea - interArea

  return interArea/unionArea

def getIoUs(df, results):
  """ Returns a list of average IoU across objects for all images """
  IoUs = []
  objCount = 0

  for i in range(len(results.pandas().xyxy)):
    img = results.pandas().xyxy[i]
    predBBcoords = getCoords(img)
    numObj = predBBcoords[0]
    avgIoU = []
    for k in range(len(numObj)):
      imgID = df.iloc[objCount].ImageID
      IoU = compute_IoU(df, imgID, predBBcoords, k, objCount)
      avgIoU.append(IoU)
      objCount += 1
    avgIoU = np.mean(avgIoU)
    IoUs.append(avgIoU)

  return IoUs

def getDetectionAcc(confidences, IoUs):
  """ Calculates and returns the detection accuracy for cofidence levels and IoU """
  confThreshold = 0.5
  IoUThreshold = 0.5

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

def runYOLOBusses():
  #Load Busses dataset
  IMAGE_ROOT = 'archive/images'
  df = pd.read_csv('archive/df.csv')

  #grab first 20 filenames
  fileNames = os.listdir('archive/images/images')
  fileNames = fileNames[0:20]

  #convert filenames to file paths
  imgs = []

  for fileName in fileNames:
    filePath = 'archive/images/images/' + fileName
    imgs.append(filePath)

  # Inference
  results = model(imgs)

  # Results
  results.print()

  # Save image results with bounding boxes and classes
  results.save()
    
  #read csv into dataframe
  df = pd.read_csv("archive/df.csv")

  confidences = getConfidences(imgs, results)
  IoUs = getIoUs(df, results)

  confAcc, IoUAcc = getDetectionAcc(confidences, IoUs)

  print("confAcc", confAcc)
  print("IoUAcc", IoUAcc)


# runYOLOSmBatch()
runYOLOBusses()