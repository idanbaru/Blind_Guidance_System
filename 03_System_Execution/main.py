import os
import cv2
import time
import base64
import depthai as dai

import pyttsx3
from gtts import gTTS
from gtts.tts import gTTSError

import queue
import threading

import utilities
import detection
import oakd_configuration
import snapshot_captioning


class OfflineTTS:
    def __init__(self):
        self.engine = pyttsx3.init()

    def setProperty(self, prop, settings):
        self.engine.setProperty(prop, settings)
    
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


# ===== GLOBALS =====
GROQ_API_KEY = snapshot_captioning.get_api_key()
detection_queue = queue.Queue()
speaking_event = threading.Event()
snapshot_caption_event = threading.Event()  # SnapCap
offline_tts = OfflineTTS()
offline_tts.setProperty('rate', 150)
online = utilities.is_online()

#intro_text = f"Initializing Blind Guidance System...\n \
#cv2 cuda support: {str(bool(cv2.cuda.getCudaEnabledDeviceCount()))}.\n \
#torch executing on: {str(torch.cuda.get_device_name())}.\n \
#depth-AI detecting camera: {str(bool(len(dai.Device.getAllAvailableDevices())))}.\n"


# ===== TEXT-TO-SPEECH WORKER =====
def speak(text: str):
    """ Speak text using gTTS (online) or pyttsx3 (offline) """
    global speaking_event, offline_tts, online
    speaking_event.set()
    if utilities.is_online() == False:
        #if online:
        #    online = False
        #    offline_tts.speak(utilities.OFFLINE_TRANSITION_TEXT)
        offline_tts.speak(text)
    else:
        try:
            #if not online:
            #    online = True
            #    gTTS(text=utilities.ONLINE_TRANSITION_TEXT, lang='en', slow=False).save('talk.mp3')
            #    os.system('mpg123 talk.mp3 > /dev/null 2>&1')
            gTTS(text=text, lang='en', slow=False).save('talk.mp3')
            os.system('mpg123 talk.mp3 > /dev/null 2>&1')
        except gTTSError:
            offline_tts.speak(text)
    time.sleep(utilities.TTS_COOLDOWN)
    speaking_event.clear()


def speaker_worker():
    """Call this in a thread, constantly speaks out loud the detections from the detection queue"""
    while True:
        detections = detection_queue.get()
        if detections is None:
            break
        # DEBUG
        print(detections)
        # DEBUG
        speak_lines = [f"{d.label} at {d.depth:.1f} meters" for d in detections]
        speak("I see: " + ", ".join(speak_lines))


# ===== SNAPSHOT CAPTIONING WORKER ===== 
def snapshot_caption_worker(shared_frame_fn):
    """Call this in a thread, periodically triggers request to Groq to shortly describe latest frame"""
    global GROQ_API_KEY, online
    while True:
        # Work in predefined intervals (default: once every 60 seconds)
        time.sleep(snapshot_captioning.SNAPSHOT_CAPTIONING_INTERVAL)
        
        if utilities.is_online() == False:
            if online:
                online = False
        
        else:
            if not online:
                online = True
            
            frame = shared_frame_fn()
            if frame is not None:
                # convert frame to base64 format (for groq query) [TODO: check if resize is needed]
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")

                # request groq for an AI description of the frame (ONLINE ONLY)
                snapshot_captioning.query_groq_with_image(base64_frame, GROQ_API_KEY)
        


# ===== MAIN =====
if __name__ == "__main__":
    # Define detection history and last frame (wrapped in a list for encapsulation)
    history = detection.DetectionHistory()
    last_frame = [None]

    def get_last_frame():
        return last_frame[0]

    # Start background threads: speaker worker and snapshot caption worker
    threading.Thread(target=speaker_worker, daemon=True).start()
    threading.Thread(target=snapshot_caption_worker, args=(get_last_frame,), daemon=True).start()

    # Connect to OAK-D Lite camera and start its pipeline (incl. video stream, neural network for detection and depth)
    pipeline = oakd_configuration.configure_oakd_camera()
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        while True:
            SPEAK_COOLDOWN = 7 # seconds
            last_spoken_time = 0
            current_detections = []
            
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            # check camera stream has a valid frame
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                last_frame[0] = frame.copy()  # TODO: lower resolution(?) and check cuda processing options

                # check if camera's nn has any detections in frame
                if inDet is not None:
                    dets = inDet.detections
                    for d in dets:
                        label = oakd_configuration.labelMap[d.label]
                        x = int((d.xmin + d.xmax) / 2 * frame.shape[1])
                        y = int((d.ymin + d.ymax) / 2 * frame.shape[0])
                        z = d.spatialCoordinates.z / 1000.0  # convert millimeters to meters (.0 to insure float result)
                        current_detections.append(detection.Detection(label=label, center=(x,y), depth=z))
                        # DEBUG
                        print(f"{label} at ({x}, {y}), depth: {z:.2f}m")
                        # DEBUG
                    
                    if current_detections and history.has_changed(current_detections):
                        now = time.time()
                        if now - last_spoken_time > SPEAK_COOLDOWN:
                            detection_queue.put(current_detections)
                            last_spoken_time = now
        
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()
                    
                
