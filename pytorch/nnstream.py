import cv2
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True
)
model.to(device)
model.eval()

transform = transforms.ToTensor()

# cap = cv2.VideoCapture('filesrc location=video.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
# print(int(cap.get(cv2.CAP_PROP_FPS)))

cap = cv2.VideoCapture("video.mp4")

# ret, frame = cap.read()

# height, width = 720, 1280# frame.shape[:2]

# fps = 0
# tau = time.time()
# smoothing = 0.9
# text_pos = (32, height - 32)

# # Parameters for drawing
# #colour = (B, G, R)
# colour = (0, 140, 255)
# thickness = 4
# fontscale = 0.5
# fontthick = 1

def gen_frames():
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
                transform = transforms.ToTensor()
                img = transform(pic).to(device) # Cast to torch.tensor() and send to GPU
                batch = [img]

                # Use the model and visualize the prediction
                prediction = model(batch)[0]

                # Load predictions to CPU for processing
                scores = np.array(prediction["scores"].cpu().detach().numpy())
                boxes = np.array(prediction["boxes"].cpu().detach().numpy())
                labels = np.array(prediction["labels"].cpu().detach().numpy())

                # Perform Non-Maximum Suppression and return list of indexes with best values
                sorted = torchvision.ops.nms(
                    boxes=prediction["boxes"], 
                    scores=prediction["scores"],
                    iou_threshold=0.08
                )

                # Rearrange predictions accordingly
                scores = np.array([scores[i] for i in sorted])
                boxes = np.array([boxes[i] for i in sorted])
                labels = np.array([labels[i] for i in sorted])

                # Perform visualising operations by casting data to the original input stream (frame)
                confidence_threshold = 0.50 # Set threshold which will be considered when sorting data
                for score, box, label in zip(scores, boxes, labels):
                    # Check for a specific label and filter out better scoring data
                    if label != 9 or score < confidence_threshold:
                        continue
                    
                    # Set the label name and configure the boxes and scores
                    label = "Vessel"
                    (xmin,ymin,xmax,ymax) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    score_txt = f"{round(score*100.0, 2)}%"
                    
                    cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), colour, thickness)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (w, h), _ = cv2.getTextSize(
                            f"{label}: " + score_txt, font, fontscale, fontthick)

                    # This block for seeing the values on detection boxes
                    # cv2.rectangle(pic, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
                    cv2.putText(pic, f"{label}: " + score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)

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