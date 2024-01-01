import cv2
import torch
# import numpy as np

def show(results, pic):
    colour = (0, 140, 255)
    thickness = 4
    fontscale = 0.5
    fontthick = 1
    
    detections = results.pandas().xyxy[0]

    for i in range(detections.shape[0]):
        if detections.iloc[i]["class"] != 8:
            continue

        xmin = int(detections.iloc[i]["xmin"])
        xmax = int(detections.iloc[i]["xmax"])
        ymin = int(detections.iloc[i]["ymin"])
        ymax = int(detections.iloc[i]["ymax"])

        score_txt = f"{round(detections.iloc[i]['confidence']*100.0, 2)}%"
        
        label = detections.iloc[i]["name"]

        cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), colour, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(
                f"{label}: " + score_txt, font, fontscale, fontthick)

        # This block for seeing the values on detection boxes
        # cv2.rectangle(pic, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
        cv2.putText(pic, f"{label}: " + score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
    
    cv2.imwrite("found.jpg", pic)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)

# input_shapes = (1, 3, 1280, 1280)
# img = cv2.imread("ship2.jpeg")
# detect = model(img, size=1280)

# show(detect, img)
# print(detect)
model_onnx = "yolov5m6"
model_trt = "yolov5m6_engine"

onnx = False
tensorrt = True

if onnx:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
    input_shapes = (1, 3, 1280, 1280)

    input = torch.randn(input_shapes, requires_grad=True, device="cuda")
    print("Exporting model to onnx format")
    torch.onnx.export(model, input, f"{model_onnx}.onnx", opset_version=12)

if tensorrt:
    from subprocess import call
    call(f"trtexec --onnx={model_onnx}.onnx --saveEngine={model_trt}.engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --noTF32 --workspace={1 << 30}".split()) # --verbose optional