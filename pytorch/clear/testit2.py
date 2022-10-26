import cv2
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit

app = Flask(__name__)

# trt_engine_path = os.path.join("ssd_engine.trt")
# model = TrtModel(trt_engine_path)

ssd300 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math="fp32")
model = ssd300.eval().to("cuda")
traced_model = torch.jit.trace(model, [torch.randn((1, 3, 800, 800)).to("cuda")])

import torch_tensorrt

trt_model = torch_tensorrt.compile(traced_model,
    inputs = [torch_tensorrt.Input((1, 3, 800, 800), dtype=torch.half)],
    enabled_precisions = {torch.half}, # Run with FP16
    workspace_size = 1 << 20
)

cap = cv2.VideoCapture("video.mp4")

def gen_frames():
    # transform = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    height, width = 720, 1280# frame.shape[:2]

    fps = 0
    tau = time.time()
    smoothing = 0.9
    text_pos = (32, height - 32)

    # Parameters for drawing
    #colour = (B, G, R)
    colour = (0, 140, 255)
    thickness = 4
    fontscale = 0.5
    fontthick = 1

    # trt_engine_path = os.path.join("ssd_engine.trt")
    # model = TrtModel(trt_engine_path)
    # shape = model.engine.get_binding_shape(0)
    # print(shape)
    # batch_size = shape[0]

    with torch.no_grad():
        while(True):
            #Capture frame-by-frame
            ret, pic = cap.read()
            
            if not ret:
                break
            else:
                # FPS calculation
                now = time.time()
                if now > tau:  # avoid div0
                    fps = fps*smoothing + 0.1/(now - tau)
                tau = now
                
                # Prepare image for neural network
                # transform = transforms.ToTensor()
                size = [800, 800]
                # print(size)
                # img = transform(cv2.resize(pic, size)).to(device) # Cast to torch.tensor() and send to GPU
                img = cv2.resize(pic, size)
                # img = np.moveaxis(img, -1, 0)
                # print(img.shape)
                batch = [img]

                prediction = trt_model(torch.from_numpy(img).half().to("cuda"))

                # Display fps
                cv2.putText(pic, str(int(fps)), \
                            text_pos, \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', pic)
                pic = buffer.tobytes()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
    
# When everything done, pack your shite and move on
cap.release()
cv2.destroyAllWindows()