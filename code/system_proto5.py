# OS utilities
from pathlib import Path
import os
import time

# Image detection and deep learning
import cv2
import torch
import numpy as np

# Utilities for groq api (replacing groq library in EOL python3.6)
import io
import json
import base64
import requests

# Camera stream
import depthai as dai

# Threads for simultaneous threads executing different tasks
import queue
import threading
import subprocess

# Text-to-Speech models (offline & online)
import pyttsx3
from gtts import gTTS
from gtts.tts import gTTSError

import socket
def is_online(host="8.8.8.8", port=53, timeout=1):
    # Tries to form a TCP connection to Google's DNS for 200ms max
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except socket.error:
        return False


# === Globals ===
detection_queue = queue.Queue()
speaking_event = threading.Event()
snapshot_caption_event = threading.Event()  # SnapCap
speak_lock = threading.Lock()
SNAPSHOT_CAPTIONING_INTERVAL = 60  # seconds
TTS_COOLDOWN_CAPTIONING = 15 # seconds
TTS_COOLDOWN_DETECTIONS = 5 # seconds

# === Load API Key ===
#TODO: set absolute path
api_key_path = str((Path(__file__).parent.parent / Path('auxiliary/config_secret.json')).resolve().absolute())
print(f"Importing groq API key from: {api_key_path}")
if not Path(api_key_path).exists():
    #import sys
    raise FileNotFoundError(f'API key not found.\n  \
                            NOTE: THE API KEY IS PRIVATE PER USER, \
                            IF YOU\'VE CLONED THIS PROJECT YOU MUST \
                            GET YOUR OWN KEY FROM: console.groq.com/keys')
with open(api_key_path) as f:
    GROQ_API_KEY = json.load(f)['GROQ_API_KEY']

# === Groq API Info ===
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

class OfflineTTS:
    def __init__(self):
        self.engine = pyttsx3.init()
    
    def setProperty(self, prop, settings):
        self.engine.setProperty(prop, settings)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


#offline_engine = pyttsx3.init()
tts = OfflineTTS()
tts.setProperty('rate', 125)
# === Utility Functions ===
def speak(text: str, caller="detection"):
    """ Speak text using gTTS (online) or pyttsx3 (offline) """
    global speaking_event, tts, TTS_COOLDOWN_DETECTIONS, TTS_COOLDOWN_CAPTIONING

    speaking_event.set()
    if is_online() == False:
        tts.speak(text)
    
    else:
        try:
            gTTS(text=text, lang='en', slow=False).save('talk.mp3')
            #os.system('mpg123 talk.mp3 > /dev/null 2>&1')
            subprocess.run(['mpg123', 'talk.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except gTTSError:
            tts.speak(text)
    
    #if caller == "detection":
    #    time.sleep(TTS_COOLDOWN_DETECTIONS)
    #else:
    #    time.sleep(TTS_COOLDOWN_CAPTIONING)
    speaking_event.clear()


def encode_image_from_cv2(image):
    _, buffer = cv2.imencode(".jpg", image) 
    return base64.b64encode(buffer).decode("utf-8")


def query_groq_with_image(base64_image):
    global speak_lock
    # The image to be sent to the model (base64 format, received in function's arguments)
    image_content = {
        "type": "image_url", 
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
    
    # The prompt to be sent alongside the image (what to do with it)
    prompt = "In a short sentence, please describe the image to a blind person."
    
    # Data of the request, includes the model selected for the task
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ]
            }
        ]
    }
    
    try:
        response = requests.post(GROQ_URL, headers=GROQ_HEADERS, json=data)
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            print(f"AI description: {text}\n")
            # TAKE LOCK
            with speak_lock:
                speak(text, caller="captioning")
            # RETURN LOCK
        else:
            print(f"GROQ_ERROR: {response.status_code}: {response.text}")
    except Exception as e:
        print("Exception:", e)


def speaker_worker():
    global speaking_event, speak_lock
    SPEAK_COOLDOWN = 2 # seconds
    last_spoken_time = 0
    while True:
        detections = detection_queue.get()
        if detections is None:
            break
        print(detections)
        
        now = time.time()
        if now - last_spoken_time > SPEAK_COOLDOWN:
            if not speaking_event.is_set():
                # ONLY EXECUTE IF LOCK IS AVAILABLE
                with speak_lock:
                    speak_lines = [f"{d.label} at {d.depth:.1f} meters" for d in detections]
                    speak("I see: " + " and ".join(speak_lines))
                    last_spoken_time = now
        
        #speak(detections)


def snapshot_caption_worker(shared_frame_fn):
    """Call this in a thread, periodically triggers Groq with latest frame"""
    while True:
        time.sleep(10)
        # if is_online():
        frame = shared_frame_fn()
        if frame is not None:
            b64_img = encode_image_from_cv2(frame)
            query_groq_with_image(b64_img)
        # end if
        time.sleep(50)


# LIST OF LABELS WHICH REQUIRES ATTENTION & FURTHER DETAILS
requires_attention = ['pothole', 'crossroad', 'bus_station']

class Detection:
    def __init__(self, label, center=(0,0), depth=0):
        self.label = label
        self.center = center
        self.depth = depth
    
    # TODO: edit magic 25 and 2 numbers depends on image resulotion
    def __eq__(self, other):
        if not isinstance(other, Detection):
            return False
        return self.label == other.label # TODO: implement the more sophisticated logic
        #return (self.label == other.label and
        #        abs(self.center[0] - other.center[0]) < 25 and
        #        abs(self.center[1] - other.center[1]) < 25 and
        #        abs(self.depth - other.depth) < 2) # allow small fluctuations (TODO: edit these magic numbers)

    @staticmethod
    def direction(x_center):
        # TODO: define relative img size
        IMG_SIZE = 640
        if x_center < IMG_SIZE / 3:
            return "left"
        elif x_center > 2*IMG_SIZE / 3:
            return "right"
        else:
            return "middle"

    # TODO: edit different cases for different labels, for example:
    # if label in requires_attention return Attention! {self.label} coming from the {direction(self.center[0])} in {self.depth} meters.
    def __repr__(self):
        return f"{self.label} at {self.center}, {self.depth:.2f}m"


class DetectionHistory:
    def __init__(self, max_len=10):
        self.history = []
        self.max_len = max_len

    def is_empty(self):
        if len(self.history) == 0:
            return True
        return False

    def has_changed(self, current_detections):
        if not self.history or self.history[-1] != current_detections:
            self.history.append(current_detections)
            if len(self.history) > self.max_len:
                self.history.pop(0)
            return True
        return False


#=========== Main Loop =============
def main():
    # Get yolo model blob file path
    nnPath = str((Path(__file__).parent / Path('models/yolov8n_openvino_2022.1_6shave.blob')).resolve().absolute())
    #nnPath = str((Path(__file__).parent / Path('yolo11n.blob')).resolve().absolute()) # OAK-D LITE SUPPORTS YOLOv5~v8 (!!!)
    print(f"Importing neural network from: {nnPath}")
    if not Path(nnPath).exists():
        raise FileNotFoundError(f'Required file/s not found')

    labelMap = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ]
    syncNN = True

    # Create pipeline and define sources and outputs
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    #detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    # Add depth pipeline (TODO: check this)
    depth = pipeline.create(dai.node.StereoDepth)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Define properties
    camRgb.setPreviewSize(640, 640)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    # Configure Mono Cameras (TODO: check this)
    for monoCam, socket in [(monoLeft, dai.CameraBoardSocket.LEFT), (monoRight, dai.CameraBoardSocket.RIGHT)]:
        monoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoCam.setBoardSocket(socket)

    # Configure StereoDepth
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB
    depth.setSubpixel(True)  # Optional: improves accuracy

    # Optional: tuning near/far range
    #depth.setDepthLowerThreshold(100)   # mm
    #depth.setDepthUpperThreshold(10000) # mm

    # Define the neural network's settings
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.input.setBlocking(False)

    # Settings for the Stereo to neural network connection (TODO: check)
    detectionNetwork.setBoundingBoxScaleFactor(0.5)  # shrink box for depth avg
    detectionNetwork.setDepthLowerThreshold(100)
    detectionNetwork.setDepthUpperThreshold(10000)

    # Enable spatial detection and sets the depth of each detection to be the average of points
    #detectionNetwork.setSpatialBoundingBoxScaleFactor(0.5)
    detectionNetwork.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.AVERAGE)


    # Linking (syncNN=True to pass video stream through the neurtal network) 
    camRgb.preview.link(detectionNetwork.input)
    if syncNN:
        detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    # Connect depth to the neural network (TODO: check)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(detectionNetwork.inputDepth)


    # Define detection history and last frame
    # TODO: add a detection class with parameters like label, amount detected, depth, center, ...
    history = DetectionHistory()
    latest_frame = [None]  # wrapped in list for closure access

    # TODO: add here the resize logic to 128x128
    def get_latest_frame():
        return latest_frame[0]

    # Start background threads
    threading.Thread(target=speaker_worker, daemon=True).start()
    threading.Thread(target=snapshot_caption_worker, args=(get_latest_frame,), daemon=True).start()

    # Connect to the OAK-D Lite device and start pipeline
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        while True:
            # DEFINITIONS
            current_detections = []
            
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            #if inRgb is None:
            #    exit("KOKO")
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                latest_frame[0] = frame.copy()  # TODO: lower resolution using CUDA to 128x128

            if inRgb is not None and inDet is not None:
                dets = inDet.detections
                for d in dets:
                    label = labelMap[d.label]
                    x = int((d.xmin + d.xmax) / 2 * frame.shape[1])
                    y = int((d.ymin + d.ymax) / 2 * frame.shape[0])
                    z = d.spatialCoordinates.z / 1000.0  # convert mm → meters
                    current_detections.append(
                        Detection(
                            label=label,
                            center=(x,y),
                            depth=z
                        )
                    )
                    print(f"{label} at ({x}, {y}), depth: {z:.2f}m")
                
                if current_detections and history.has_changed(current_detections):
                    detection_queue.put(current_detections)
                
                #labels = [labelMap[d.label] for d in dets]
                #label_string = ", ".join(labels)
   
                #if label_string is not "" and history.has_changed(label_string):
                #    detection_queue.put(f"I see: {label_string}")


            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
