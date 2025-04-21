import os
import time
from datetime import datetime
import argparse
import logging
import utilities
from pathlib import Path

# Image detection and deep learning
import cv2
import numpy as np
import detection

# Utilities for groq api (replacing groq library in EOL python3.6)
import base64
import snapshot_captioning

# Camera stream
import depthai as dai
import oakd_configuration

# Threads for simultaneous tasks
import queue
import threading
import subprocess

# Text-to-Speech models
import pyttsx3
from gtts import gTTS
from gtts.tts import gTTSError

# === Globals ===
detection_queue = utilities.RingBufferQueue(maxsize=5)
speaking_event = threading.Event()
snapshot_caption_event = threading.Event()
speak_lock = threading.Lock()
latest_frame = [None]
is_online = utilities.is_online()
TTS_COOLDOWN_CAPTIONING = 15
TTS_COOLDOWN_DETECTIONS = 5
SYSTEM_LANGUAGE = 'en'


class OfflineTTS:
    def __init__(self):
        self.engine = pyttsx3.init()

    def setProperty(self, prop, settings):
        self.engine.setProperty(prop, settings)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


tts = OfflineTTS()
tts.setProperty('rate', 125)


def speak(text: str, language='en', caller="detection"):
    global speaking_event, tts, TTS_COOLDOWN_DETECTIONS, TTS_COOLDOWN_CAPTIONING
    if language == 'he':
        language = 'iw'
    speaking_event.set()
    try:
        gTTS(text=text, lang=language, slow=False).save('talk.mp3')
        subprocess.run(['mpg123', 'talk.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except gTTSError:
        tts.speak(text)
    speaking_event.clear()


def speaker_worker():
    global speaking_event, speak_lock, SYSTEM_LANGUAGE
    SPEAK_COOLDOWN = 2
    DETECTION_TIMEOUT = 3
    last_spoken_time = 0
    speaker_controler = detection.SpeechController()
    while True:
        item = detection_queue.get()
        if item is None:
            break

        detections, insert_time = item
        now = time.time()

        if now - insert_time > DETECTION_TIMEOUT:
            logging.info("[Info] Skipped Outdated Detections")
            continue

        if now - last_spoken_time > SPEAK_COOLDOWN:
            if not speaking_event.is_set():
                with speak_lock:
                    now = time.time()
                    if now - insert_time > DETECTION_TIMEOUT:
                        logging.info("[Info] Skipped Outdated Detection (post-lock)")
                        continue
                    #text = "I see: " + " and ".join([d.__repr__() for d in detections])
                    text = speaker_controler.summarize_detections(detections=detections)
                    if text is not None:
                        logging.info(f"[Speak] {text}")
                        speak(text)
                        last_spoken_time = now


def snapshot_caption_worker(shared_frame_fn):    
    def encode_image_from_cv2(image):
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    global speak_lock, SYSTEM_LANGUAGE, is_online
    GROQ_API_KEY = snapshot_captioning.get_api_key()
    while True:
        start_time = time.time()
        time.sleep(10)
        frame = shared_frame_fn()
        if frame is not None:
            b64_frame = encode_image_from_cv2(frame)
            if (not utilities.is_online()):
                logging.info(f"[Caption] failed to connect Groq, skipping...")
                if is_online:
                    is_online = False
                    with speak_lock:
                        speak(text="system is offline")
                time.sleep(10)
                continue
            else:
                if not is_online:
                    is_online = True
                    with speak_lock:
                        speak(text="system back online!")
            frame_caption = snapshot_captioning.query_groq_with_image(
                base64_image=b64_frame,
                api_key=GROQ_API_KEY,
                language=SYSTEM_LANGUAGE
            )
            logging.info(f"[Caption] {frame_caption}")
            with speak_lock:
                speak(text=frame_caption, language=SYSTEM_LANGUAGE, caller="captioning")
        while time.time() - start_time < 50:
            time.sleep(5)


def main():
    global latest_frame

    args = parser.parse_args()

    # === Create session directory ===
    session_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    session_dir = os.path.join("sessions", session_time)
    os.makedirs(session_dir, exist_ok=True)

    # === Setup logging ===
    log_path = os.path.join(session_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Session started")

    # Model path
    path = None
    if args.mode:
        if args.mode == 'indoor':
            logging.info("Loading indoor model")
            path = str((Path(__file__).parent.parent / 'auxiliary/models/yolov8n_indoor_5shave.blob').resolve())
        elif args.mode == 'outdoor':
            logging.info("Loading outdoor model")
            path = str((Path(__file__).parent.parent / 'auxiliary/models/yolov8n_outdoor_5shave.blob').resolve())
        else:
            logging.error("Invalid mode. Use 'indoor' or 'outdoor'.")
            exit(1)

    pipeline = oakd_configuration.configure_oakd_camera(nnPath=path, mode=args.mode)
    history = detection.DetectionHistory()

    def get_latest_frame():
        return latest_frame[0]

    threading.Thread(target=speaker_worker, daemon=True).start()
    threading.Thread(target=snapshot_caption_worker, args=(get_latest_frame,), daemon=True).start()

    record = args.record
    clip_duration = 30
    fps = 12
    writer = None
    start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        while True:
            current_detections = []
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                raw_frame = frame.copy()
                latest_frame[0] = raw_frame.copy()

                if inDet is not None:
                    dets = inDet.detections
                    for d in dets:
                        # TODO: for Indoor mode, remove irrelevant classes.
                        labelMap = oakd_configuration.get_label_map(args.mode)
                        label = labelMap[d.label]
                        x = int((d.xmin + d.xmax) / 2 * frame.shape[1])
                        y = int((d.ymin + d.ymax) / 2 * frame.shape[0])
                        z = d.spatialCoordinates.z / 1000.0
                        
                        current_detection = detection.Detection(label=label, center=(x, y), depth=z)
                        
                        location = current_detection.direction()
                        text = f"{label} ({z:.2f}m) {location}"
                        
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, text, (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # # Convert normalized bbox to pixel coordinates
                        # top_left = (int(d.xmin * frame.shape[1]), int(d.ymin * frame.shape[0]))
                        # bottom_right = (int(d.xmax * frame.shape[1]), int(d.ymax * frame.shape[0]))

                        # # Draw rectangle based on detection bounding box
                        # cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                                        
                        
                        # cv2.putText(frame, text, (x - 10, int(d.ymin * frame.shape[0]) - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (255, 255, 255), 2, cv2.LINE_AA)

                        current_detections.append(
                            current_detection
                        )

                    if current_detections and history.has_changed(current_detections):
                        logging.info(f"[Detections] {current_detections}")
                        detection_queue.put((current_detections, time.time()))

                if record:
                    if writer is None:
                        h, w = frame.shape[:2]
                        filename = datetime.now().strftime("%d%m%Y_%H%M%S") + ".mp4"
                        filepath = os.path.join(session_dir, filename)
                        logging.info(f"[Video] Recording started: {filename}")
                        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
                        start_time = time.time()

                    writer.write(frame)

                    if time.time() - start_time > clip_duration:
                        writer.release()
                        logging.info(f"[Video] Recording saved.")
                        writer = None

            time.sleep(0.002)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Blind Guidance System",
                                     description="The system assists visually impaired users via real-time object detection and audio feedback.")
    parser.add_argument('-m', '--mode', default='indoor', help='Model to use: indoor or outdoor')
    parser.add_argument('-r', '--record', action='store_true', help='Enable video recording')

    try:
        main()
    except KeyboardInterrupt:
        logging.info("[Main] Stopping gracefully...")
        exit(0)
