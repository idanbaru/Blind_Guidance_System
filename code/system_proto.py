# OS utilities
from pathlib import Path
import os
import sys
import time

# Image detection and deep learning
import cv2
import torch
import numpy as np

#TODO: check openai API
# import openai

# Camera stream
import depthai as dai

# Threads for simultaneous threads executing different tasks
import threading

# Text-to-Speech models (offline & online)
import pyttsx3
from gtts import gTTS
from gtts.tts import gTTSError

release = False
confidence_level = 0.4
speak_flag = True
text = 'Empty'
old_text = ''


def speak(text : str):
    # Make sure only main thread speaks the initializations:
    #if threading.current_thread() != threading.main_thread():
    #    return
    
    # Speak with gTTS (if online), otherwise speak with pyttsx3 (espeak)
    try:
        gTTS(text=text, lang='en', slow=False).save('talk.mp3')
        os.system('mpg123 talk.mp3')
    except gTTSError as e:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def speakText():
    global speak_flag
    global text
    while True:
        if speak_flag:
            speak(text)
            speak_flag = False


#if __name__ == "__main__":
if release and threading.current_thread() == threading.main_thread():
    speak(f"Initializing Blind Guidance System...\n \
            cv2 cuda support: {str(bool(cv2.cuda.getCudaEnabledDeviceCount()))}.\n \
            torch executing on: {str(torch.cuda.get_device_name())}.\n \
            depth-AI detecting camera: {str(bool(len(dai.Device.getAllAvailableDevices())))}.\n"
        )
    speak("Starting system!")

# Start the Text-to-Speech thread
if release and threading.current_thread() == threading.main_thread():
    speak("Starting the Text-to-Speech thread...")
tts_thread = threading.Thread(target=speakText, daemon=True)
tts_thread.start()

# Load YOLO model
if release and threading.current_thread() == threading.main_thread():
    speak("Loading YOLO model...")
nnPath = str((Path(__file__).parent / Path('models/yolov8n_openvino_2022.1_6shave.blob')).resolve().absolute())
if not Path(nnPath).exists():
    raise FileNotFoundError("Required YOLO model not found")

labelMap = [
    "person",
    "traffic light",
    "bench",
    "bird",
    "cat",
    "dog",
    "backpack",
    "umbrella",
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "carrot",
    "pizza",
    "chair",
    "bed",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink", 
    "refrigerator",
    "book", 
    "clock",
    "scissors",
    "toothbrush"
]

# Create pipeline
if release and threading.current_thread() == threading.main_thread():
    speak("Creating Pipeline to OAK-D Lite Camera...")
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

detectionNetwork.setConfidenceThreshold(confidence_level)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Connect to device
if release and threading.current_thread() == threading.main_thread():   
    speak("Connecting to OAK-D Lite camera...")
with dai.Device(pipeline) as device:
    max_size = 1
    qRgb = device.getOutputQueue(name="rgb", maxSize=max_size, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=max_size, blocking=False)

    frame = None
    detections = []
    previous_label = ""
    
    # Starting main loop
    if release and threading.current_thread() == threading.main_thread():
        speak("Entering main loop!")
    while True:
        #inRgb = qRgb.get()
        #inDet = qDet.get()
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()
        frame = inRgb.getCvFrame() if inRgb else None
        detections = inDet.detections if inDet else []
            
        for detection in detections:
            #bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            label = labelMap[detection.label] if detection.label < len(labelMap) else "Unknown"
            confidence = detection.confidence

            if confidence >= confidence_level and label != previous_label:
                text = f"Detected {label} with {int(confidence * 100)} percent confidence."
                speak_flag = True
                previous_label = label

            if speak_flag:
                speakText()
                speak_flag = False

        # Display detection (DEBUG)
        #'''
        if frame is not None:
            for detection in detections:
                bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                bbox = (int(bbox[0] * frame.shape[1]), int(bbox[1] * frame.shape[0]),
                        int(bbox[2] * frame.shape[1]), int(bbox[3] * frame.shape[0]))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("DepthAI Camera", frame)
            if cv2.waitKey(1) == ord('q'):
                break

cv2.destroyAllWindows()
        #'''
