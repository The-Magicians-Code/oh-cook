import torch
import time
import numpy as np
import torch.backends.cudnn as cudnn
from subprocess import call

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
cudnn.benchmark = True

# load SSD model pretrained on COCO from Torch Hub
precision = 'fp32'
ssd300 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

# Next, we run object detection
model = ssd300.eval().to("cuda")

###########################Benchmark
# Helper function to benchmark the model
def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc, pred_label = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    # print(model(input_data))
    print("Input shape:", input_data.size())
    print("Output location prediction size:", pred_loc.size())
    print("Output label prediction size:", pred_label.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))
    print('Average FPS: %0.0f' % (1.0 / np.mean(timings)))
    print(pred_loc, pred_label)
############################Benchmark

input_shapes = (1, 3, 300, 300)

import cv2
pic = cv2.imread("ship2.jpeg")
size = (300, 300)
img = np.transpose(cv2.resize(pic, size), (2, 0, 1))
img = torch.Tensor(img)
pred_loc, pred_label = model(img.unsqueeze(0).to("cuda"))
print(f"loc:\n{pred_loc}\nlabel:\n{pred_label}")
print(f"loc.shape:\n{pred_loc.shape}\nlabel.shape:\n{pred_label.shape}")

onnx = False
tensorrt = False
model_onnx = "ssd2.onnx"
model_trt = "ssd_engine2.trt"
# benchmark(model, input_shape=input_shapes, nruns=50)
# print(model)
if onnx:
    output = torch.randn(input_shapes, requires_grad=True, device="cuda")
    print("Exporting model to onnx format")
    torch.onnx.export(model, output, model_onnx)

if tensorrt:
    call(f"trtexec --onnx={model_onnx} --saveEngine={model_trt} --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --workspace={1 << 30} --verbose".split())