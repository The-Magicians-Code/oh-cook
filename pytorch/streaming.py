import random
import time

from flask import Flask, render_template, Response, jsonify, request
import cv2

import numpy
# import tensorflow as tf

app = Flask(__name__)

# USB camera / video
# camera = cv2.VideoCapture('filesrc location=video.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
camera = cv2.VideoCapture("video.mp4")
video_fps = camera.get(cv2.CAP_PROP_FPS)/1000.0
print(int(camera.get(cv2.CAP_PROP_FPS)))

def gen_frames():
    # FPS variables for calculation
    fps = 0
    tau = time.time()
    smoothing = 0.9
    while True:
        #time.sleep(video_fps)
        
        # FPS calculation
        now = time.time()
        if now > tau:  # avoid div0
            fps = fps*smoothing + 0.1/(now - tau)
        tau = now
        # print(int(fps))
        
        # Capture frame-by-frame
        success, picture = camera.read()  # read the camera frames

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', picture)
            picture = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + picture + b'\r\n')  # concat frame one by one and show result


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