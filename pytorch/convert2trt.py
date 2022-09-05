import subprocess

subprocess.call("trtexec --onnx=fasterrcnn.onnx --saveEngine=fasterrcnn_engine.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16".split())

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torchvision.transforms as transforms
import cv2

f = open("fasterrcnn_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

import numpy as np

BATCH_SIZE = 1
# Prep the image for feeding into the network
transform = transforms.ToTensor()
i = cv2.imread("./ship2.jpeg") # Read as float32
img = transform(i).to("cuda")
input_batch = [img]
# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, 1000], dtype = np.float16)

# allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output

print("Warming up...")

pred = predict(input_batch)

print("Done warming up!")

import time
# Speed benchmark
times = []
for i in range(20):
    start_time = time.time()
    preds = predict(input_batch)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1/mean_delta
print("Average(sec):{}, fps:{}".format(mean_delta,fps))