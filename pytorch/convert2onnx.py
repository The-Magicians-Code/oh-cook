import torchvision
import torchvision.models as models
import torch
import torch.onnx
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True
)

# model.to(device)
# Model to evaluation mode
model.eval()

BATCH_SIZE = 1

dummy_input=torch.randn(BATCH_SIZE, 3, 800, 800)

# export the model to ONNX
torch.onnx.export(model, dummy_input, "fasterrcnn.onnx", verbose=False, opset_version = 11)

import os

os._exit(0) # Shut down all kernels so TRT doesn't fight with PyTorch for GPU memory