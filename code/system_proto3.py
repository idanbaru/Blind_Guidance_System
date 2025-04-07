# OS utilities
from pathlib import Path
import os
import sys
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
import threading
import queue

# Text-to-Speech models (offline & online)
import pyttsx3
from gtts import gTTS
from gtts.tts import gTTSError


# === Globals ===
detection_queue = queue.Queue()
speaking_event = threading.Event()
snapshot_caption_event = threading.Event()  # SnapCap
SNAPSHOT_CAPTIONING_INTERVAL = 60  # seconds

# === Load API Key ===
with open('../auxiliary/config_secret.json') as f:
    GROQ_API_KEY = json.load(f)['GROQ_API_KEY']

# === Groq API Info ===
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

# === Utility Functions ===
def speak(text: str):
    """ Speak text using gTTS (online) or pyttsx3 (offline) """
    global speaking_event
    speaking_event.set()
    try:
        gTTS(text=text, lang='en', slow=False).save('talk.mp3')
        os.system('mpg123 talk.mp3')
    except gTTSError:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    speaking_event.clear()


def encode_image_from_cv2(image):
    _, buffer = cv2.imencode(".jpg", image) 
    return base64.b64encode(buffer).decode("utf-8")


def query_groq_with_image(base64_image):
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
            speak(text)
        else:
            print(f"GROQ_ERROR: {response.status_code}: {response.text}")
    except Exception as e:
        print("Exception:", e)


def speaker_worker():
    while True:
        detections = detection_queue.get()
        if detections is None:
            break
        speak(detections)


def snapshot_caption_worker(shared_frame_fn):
    """Call this in a thread, periodically triggers Groq with latest frame"""
    while True:
        time.sleep(SNAPSHOT_CAPTIONING_INTERVAL)
        frame = shared_frame_fn()
        if frame is not None:
            b64_img = encode_image_from_cv2(frame)
            query_groq_with_image(b64_img)

#class Detection:
#    def __init__(self, img, label, center=(0,0), depth=0, )

class DetectionHistory:
    def __init__(self, max_len=10):
        self.history = []
        self.max_len = max_len


def has_changed(self, current):
    if not self.history or self.history[-1] != current:
        self.history.append(current)
        if len(self.history) > self.max_len:
            self.history.pop(0)
        return True
    return False


#=========== Main Loop =============
def main():
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

    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath("path_to_blob.blob")  # change to your actual path
    detectionNetwork.input.setBlocking(False)

    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    history = DetectionHistory()
    latest_frame = [None]  # wrapped in list for closure access

    # TODO: add here the resize logic to 128x128
    def get_latest_frame():
        return latest_frame[0]

    # Start background threads
    threading.Thread(target=speaker_worker, daemon=True).start()
    threading.Thread(target=snapshot_caption_worker, args=(get_latest_frame,), daemon=True).start()

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        while True:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                latest_frame[0] = frame.copy()

            if inDet is not None:
                dets = inDet.detections
                labels = [labelMap[d.label] for d in dets]
                label_string = ", ".join(labels)

                if history.has_changed(label_string):
                    detection_queue.put(f"I see: {label_string}")

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
