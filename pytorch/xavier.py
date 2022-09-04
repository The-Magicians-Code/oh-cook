# This is the current prototype which works on Xavier
# Author: Tanel Treuberg

import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True
)

model.to(device)
# Model to evaluation mode
model.eval()
# print(model)

# Prep the image for feeding into the network
transform = transforms.ToTensor()
i = cv2.imread("./ship2.jpeg") # Read as float32
img = transform(i).to(device)

i8 = torch.from_numpy(i).permute(2, 0, 1) # Image to uint8, channels first
print(i8.shape)

# For drawing at the minute
# img_display = read_image("./ship2.jpeg")
# Apply inference preprocessing transforms
batch = [img]

# Use the model and visualize the prediction
prediction = model(batch)[0]

scores = prediction["scores"]
boxes = prediction["boxes"]
labels = prediction["labels"]

sorted = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=0.03)

scores = [scores[i] for i in sorted]
boxes = torch.from_numpy(np.array([boxes[i].cpu().detach().numpy() for i in sorted]))
labels = [labels[i] for i in sorted]

labels_and_scores = [f"Vessel:{score*100:.2f}%" for label, score in zip(labels, scores) 
                                                                    if label == 9]

box = draw_bounding_boxes(i8, boxes=boxes,
                          labels=labels_and_scores,
                          colors="red",
                          width=4)
im = to_pil_image(box.detach())
im.save("pred.jpg")
# im.show()
