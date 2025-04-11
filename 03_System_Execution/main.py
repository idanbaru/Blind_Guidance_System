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
import snapshot_captioning

# Camera stream
import depthai as dai
import oakd_configuration

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
SYSTEM_LANGUAGE = 'he'

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
def speak(text: str, language='en', caller="detection"):
    """ Speak text using gTTS (online) or pyttsx3 (offline) """
    global speaking_event, tts, TTS_COOLDOWN_DETECTIONS, TTS_COOLDOWN_CAPTIONING
    if language == 'he':
        language = 'iw'  # for some reason gtts addresses hebrew as 'iw'
    # TODO: check if speaking_event is necessary because implementation with lock
    speaking_event.set()
    if is_online() == False:
        tts.speak(text)
    
    else:
        try:
            gTTS(text=text, lang=language, slow=False).save('talk.mp3')
            #os.system('mpg123 talk.mp3 > /dev/null 2>&1')
            subprocess.run(['mpg123', 'talk.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except gTTSError:
            tts.speak(text)
    speaking_event.clear()


def speaker_worker():
    global speaking_event, speak_lock
    SPEAK_COOLDOWN = 2 # seconds
    last_spoken_time = 0
    while True:
        # TODO: (detections, insert_time) = queue.get()
        detections = detection_queue.get()
        if detections is None:
            break
        print(detections)
        
        now = time.time()
        # TODO: check here if time.now() - insert_time > 3 seconds then clear queue and continue (do not speak older detections)
        if now - last_spoken_time > SPEAK_COOLDOWN:
            if not speaking_event.is_set():
                # ONLY EXECUTE IF LOCK IS AVAILABLE
                with speak_lock:
                    # TODO: check here if time.now() - insert_time > 3 seconds then clear queue and continue (do not speak older detections)
                    speak_lines = [f"{d.label} at {d.depth:.1f} meters" for d in detections]
                    speak("I see: " + " and ".join(speak_lines))
                    last_spoken_time = now
        
        #speak(detections)


def snapshot_caption_worker(shared_frame_fn):
    """Call this in a thread, periodically triggers Groq with latest frame"""
    def encode_image_from_cv2(image):
        _, buffer = cv2.imencode(".jpg", image) 
        return base64.b64encode(buffer).decode("utf-8")
    global speak_lock, SYSTEM_LANGUAGE
    GROQ_API_KEY = snapshot_captioning.get_api_key()
    while True:
        time.sleep(10)
        # if is_online():
        frame = shared_frame_fn()
        if frame is not None:
            b64_frame = encode_image_from_cv2(frame)
            # TODO: try: here and catch HTML exception after 3 failed requests
            frame_caption = snapshot_captioning.query_groq_with_image(
                base64_image=b64_frame,
                api_key=GROQ_API_KEY,
                language=SYSTEM_LANGUAGE
            )
            print(frame_caption)
            with speak_lock:
                speak(text=frame_caption, language=SYSTEM_LANGUAGE, caller="captioning")
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
    pipeline = oakd_configuration.configure_oakd_camera()

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

            if inRgb is not None:   # AN RGB FRAME WAS CAPTURED
                frame = inRgb.getCvFrame()
                latest_frame[0] = frame.copy()  # TODO: lower resolution using CUDA to 128x128

                if inDet is not None:   # THE NN DETECTED OBJECTS IN THE FRAME
                    dets = inDet.detections
                    for d in dets:
                        label = oakd_configuration.labelMap[d.label]
                        x = int((d.xmin + d.xmax) / 2 * frame.shape[1])
                        y = int((d.ymin + d.ymax) / 2 * frame.shape[0])
                        z = d.spatialCoordinates.z / 1000.0  # convert millimeters to meters
                        current_detections.append(
                            Detection(
                                label=label,
                                center=(x,y),
                                depth=z
                            )
                        )               
                    
                    if current_detections and history.has_changed(current_detections):
                        print(f"{label} at ({x}, {y}), depth: {z:.2f}m")
                        detection_queue.put(current_detections)
                        # TODO: add time stamp for each detection and insert it as tuple (current_detections, current_time)

            # IMPORTANT: DO NOT REMOVE THIS PART
            # There must be a certain pause between each frame handling to allow the 2 worker threads (speaking the detections and the image captioning) time to execute
            # This is crucial to prevent overloading of the CPU and allow the speaker worker to empty the detection queue in a reasonable time
            time.sleep(0.005)   # 5 milliseconds



if __name__ == "__main__":
    main()
