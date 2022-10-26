import torchvision.models as models
import torch
import torch.onnx
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True, 
    min_size=800, 
    max_size=800
)

# model.to(device)
# Model to evaluation mode
model.eval()

BATCH_SIZE = 1

dummy_input = torch.randn((BATCH_SIZE, 3, 800, 800))#, device = device)

# export the model to ONNX
torch.onnx.export(model, dummy_input, "fasterrcnn.onnx", verbose=True, do_constant_folding=True, opset_version = 11)

import os

os._exit(0) # Shut down all kernels so TRT doesn't fight with PyTorch for GPU memory


# model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True,min_size=800,max_size=1333) 
# image=cv2.imread("test.jpg") 
# image=cv2.resize(image,(1333,800))
# image1 = Image.fromarray(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)) 
# image_tensor=to_tensor(image1) model.eval() 
# onnx_io = io.BytesIO() 
# torch.onnx.export(model, [image_tensor], "faster_rcnn.onnx",do_constant_folding=True, opset_version=_onnx_opset_version)