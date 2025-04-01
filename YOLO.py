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

import torch
import numpy as np
import torchvision.io as io
#import selectivesearch
#import torch_snippets
#from torch_snippets import *
from torchvision import torch_snippets

"""
#get class names
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  pretrained=True)
results = model(img)
classes = results.names

"""

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

print("Hello World")
results.print()
classes = results.names
print(classes)

im = np.random.rand(100, 100)
#show(im)

torch_snippets.show(im)
