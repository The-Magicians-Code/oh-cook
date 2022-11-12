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

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self, x: np.ndarray, batch_size=2):
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]

# trt_engine_path = os.path.join("ssd_engine.trt")
# model = TrtModel(trt_engine_path)
# shape = model.engine.get_binding_shape(0)
# print(shape)
# batch_size = shape[0]
# data = np.random.randint(0,255,(batch_size,*shape[1:]))/255
# result = model(data, batch_size)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
#     pretrained=True
# )
# model.to(device)
# model.eval()

# cap = cv2.VideoCapture('filesrc location=video.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
# print(int(cap.get(cv2.CAP_PROP_FPS)))

trt_engine_path = os.path.join("ssd_engine2.trt")
model = TrtModel(trt_engine_path)

cap = cv2.VideoCapture("video.mp4")

def gen_frames():
    # transform = transforms.ToTensor()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

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
    shape = model.engine.get_binding_shape(0)
    print(shape)
    batch_size = shape[0]
    # data = np.random.randint(0,255,(batch_size,*shape[1:]))/255
    # result = model(data, batch_size)

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
                size = shape[2:]
                img = np.transpose(cv2.resize(pic, size), (2, 0, 1))
                print(img.shape)

                loc, label = model(img, batch_size)
                loc = torch.Tensor(loc.reshape((1, 4, 8732)))
                label = torch.Tensor(label.reshape((1, 81, 8732)))

                results = utils.decode_results([loc, label])
                # best_results = [utils.pick_best(result, 0.40) for result in results]
                print(results)

                # # Load predictions to CPU for processing
                # scores = np.array(prediction["scores"].cpu().detach().numpy())
                # boxes = np.array(prediction["boxes"].cpu().detach().numpy())
                # labels = np.array(prediction["labels"].cpu().detach().numpy())

                # # Perform Non-Maximum Suppression and return list of indexes with best values
                # sorted = torchvision.ops.nms(
                #     boxes=prediction["boxes"], 
                #     scores=prediction["scores"],
                #     iou_threshold=0.08
                # )

                # # Rearrange predictions accordingly
                # scores = np.array([scores[i] for i in sorted])
                # boxes = np.array([boxes[i] for i in sorted])
                # labels = np.array([labels[i] for i in sorted])

                # # Perform visualising operations by casting data to the original input stream (frame)
                # confidence_threshold = 0.50 # Set threshold which will be considered when sorting data
                # for score, box, label in zip(scores, boxes, labels):
                #     # Check for a specific label and filter out better scoring data
                #     if label != 9 or score < confidence_threshold:
                #         continue
                    
                #     # Set the label name and configure the boxes and scores
                #     label = "Vessel"
                #     (xmin,ymin,xmax,ymax) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                #     score_txt = f"{round(score*100.0, 2)}%"
                    
                #     cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), colour, thickness)
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     (w, h), _ = cv2.getTextSize(
                #             f"{label}: " + score_txt, font, fontscale, fontthick)

                #     # This block for seeing the values on detection boxes
                #     # cv2.rectangle(pic, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
                #     cv2.putText(pic, f"{label}: " + score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)

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
                # Display the result
                # cv2.imshow('black and white', pic)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

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