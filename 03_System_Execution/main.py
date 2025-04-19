# Utilities
import os
import time
import utilities
from pathlib import Path

# Image detection and deep learning
import cv2
#import torch
import numpy as np
import detection

# Utilities for groq api (replacing groq library in EOL python3.6)
import base64
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


# === Globals ===
#detection_queue = queue.Queue(maxsize=5)
detection_queue = utilities.RingBufferQueue(maxsize=5)
speaking_event = threading.Event()
snapshot_caption_event = threading.Event()  # SnapCap
speak_lock = threading.Lock()
SNAPSHOT_CAPTIONING_INTERVAL = 60  # seconds
TTS_COOLDOWN_CAPTIONING = 15 # seconds
TTS_COOLDOWN_DETECTIONS = 5 # seconds
SYSTEM_LANGUAGE = 'en'


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
    if utilities.is_online() == False:
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
    global speaking_event, speak_lock, SYSTEM_LANGUAGE
    SPEAK_COOLDOWN = 2 # seconds
    DETECTION_TIMOUT = 3 # seconds
    last_spoken_time = 0
    while True:
        # TODO: (detections, insert_time) = queue.get()
        item = detection_queue.get()
        if item is None:
            break
        
        detections, insert_time = item
        now = time.time()
        
        if now - insert_time > DETECTION_TIMOUT:
            print("[Info] Skipped Outdated Detections")
            continue
        
        if now - last_spoken_time > SPEAK_COOLDOWN:
            if not speaking_event.is_set():
                # ONLY EXECUTE IF LOCK IS AVAILABLE
                with speak_lock:
                    now = time.time()
                    if now - insert_time > DETECTION_TIMOUT:
                        print("[Info] Skipped Outdated Detection (post-lock)")
                        continue
                    #speak_lines = [f"{d.label} at {d.depth:.1f} meters" for d in detections]
                    text = "I see: " + " and ".join([d.__repr__() for d in detections])
                    print(text) # TODO: add log functionality to log everything that the system speaks
                    speak(text) # TODO: uncomment this
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


#=========== Main Loop =============
def main():
    pipeline = oakd_configuration.configure_oakd_camera()

    # Define detection history and last frame
    # TODO: add a detection class with parameters like label, amount detected, depth, center, ...
    history = detection.DetectionHistory()
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
                            detection.Detection(
                                label=label,
                                center=(x,y),
                                depth=z
                            )
                        )               
                    
                    if current_detections and history.has_changed(current_detections):
                        #print(f"{label} at ({x}, {y}), depth: {z:.2f}m")
                        print(current_detections)
                        detection_queue.put((current_detections, time.time()))

            # IMPORTANT: DO NOT REMOVE THIS PART
            # There must be a certain pause between each frame handling to allow the 2 worker threads (speaking the detections and the image captioning) time to execute
            # This is crucial to prevent overloading of the CPU and allow the speaker worker to empty the detection queue in a reasonable time
            time.sleep(0.005)   # 5 milliseconds



if __name__ == "__main__":
    main()
