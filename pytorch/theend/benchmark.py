# import cv2
import time
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
model_RT = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s6_engine.engine") # This line is important since it contains RT execution
model.eval().to("cuda")
model_RT.eval().to("cuda")

def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000, verbose=False):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            outs = model(input_data, size=1280)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            outs = model(input_data, size=1280)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if not i % 10 and verbose:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))
    print('Average FPS: %0.0f' % (1.0 / np.mean(timings)))

benchmark(model, input_shape=(1, 3, 1280, 1280), dtype="fp32")
print("[o] Model without RT")
benchmark(model_RT, input_shape=(1, 3, 1280, 1280), dtype="fp16")
print("[o] Model optimised with RT @fp16")
