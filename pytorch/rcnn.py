import torch
import torchvision
from torchvision.io.image import read_image
#from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True#weights=weights
)

# Model to evaluation mode
model.eval()
# print(model)

# Initialize the inference transforms
# preprocess = weights.transforms()

img = read_image("./ship2.jpeg")
# print(img.shape)

img_mod = torch.tensor(img.detach().numpy() / 255.0, dtype=torch.float32)
# Apply inference preprocessing transforms
batch = [img_mod]#preprocess(img)]
# print(batch[0].shape)

# Use the model and visualize the prediction
prediction = model(batch)[0]
#print(prediction)
# print(weights.meta["categories"])
_scores = prediction["scores"].detach().numpy()
_boxes = prediction["boxes"].detach().numpy()
_vessels = prediction["labels"].detach().numpy()

scores, boxes, labels = ([], [], [])
for box, score, label in zip(_boxes, _scores, _vessels):
    if score > 0.5:
        boxes.append(box)
        scores.append(score)
        labels.append(label)

scores = torch.tensor(np.array(scores))
boxes = torch.tensor(np.array(boxes))
labels = torch.tensor(np.array(labels))
# print(scores, boxes, labels)

# labels = [weights.meta["categories"][i] for i in labels]# prediction["labels"]]

labels_and_scores = [f"{label}:{score*100:.2f}%" for label, score in zip(labels, scores)]

# print(labels_and_scores)
box = draw_bounding_boxes(img, boxes=boxes,
                          labels=labels_and_scores,
                          colors="red",
                          width=4)
im = to_pil_image(box.detach())
im.save("pred.jpg")
# im.show()
