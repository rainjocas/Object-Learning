"""
Note: in order to get class names (to be used according to the labels the detectors output), you can use:
classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
For Pytorch’s Faster R-CNN and 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  pretrained=True)
results = model(img)
classes = results.names
For TorchHub’s YOLO. Note that in both cases, you get a dictionary as a result.
"""