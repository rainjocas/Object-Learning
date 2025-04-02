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
import pandas as pd
import torch
import numpy as np
import torchvision.io as io
#import selectivesearch
#import torch_snippets
#from torch_snippets import *
#import torch_snippets
import cv2

"""
#get class names
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  pretrained=True)
results = model(img)
classes = results.names

"""

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Batch of images
images = ["Media/Crows.jpg", "Media/Dog Park.jpg",
          "Media/Petting Zoo.jpg", "Media/Street.jpg",
          "Media/Cars Movie.png"]

# Inference
results = model(images)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

print(results)

print("Hello World")
# results.print()
# classes = results.names
# print(classes)


#torch_snippets.show(im)

#Load Busses dataset
IMAGE_ROOT = 'archive/images'
df = pd.read_csv('archive/df.csv')
print("First 5 records:", df.head())