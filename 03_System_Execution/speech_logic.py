from collections import Counter, defaultdict
import time

URGENT_CLASSES = {"crosswalk", "stairs", "bus"}
CASUAL_CLASSES = {"person", "car"}
CASUAL_THRESHOLD = 3
CASUAL_TIMEOUT = 20  # seconds before repeating casual summary
URGENT_TIMEOUT = 10  # seconds before repeating urgent warning

class SpeechController:
    def __init__(self):
        self.last_urgent_spoken = {}  # label -> timestamp
        self.last_casual_spoken = 0  # timestamp

    def summarize_detections(self, detections):
        """Return a spoken string based on detection priority and frequency"""
        now = time.time()
        urgent_lines = []
        casual_counter = Counter()

        for d in detections:
            if d.label in URGENT_CLASSES:
                if (now - self.last_urgent_spoken.get(d.label, 0)) > URGENT_TIMEOUT:
                    urgent_lines.append(f"{d.label} ahead at {d.depth:.1f} meters")
                    self.last_urgent_spoken[d.label] = now
            elif d.label in CASUAL_CLASSES:
                casual_counter[d.label] += 1

        speech_lines = []

        # Urgent always gets priority
        if urgent_lines:
            speech_lines.append("Attention: " + ", ".join(urgent_lines))
        
        # Add casual if timeout has passed and not already speaking urgent
        if not urgent_lines and (now - self.last_casual_spoken) > CASUAL_TIMEOUT and casual_counter:
            casual_descriptions = []
            for label, count in casual_counter.items():
                if count >= CASUAL_THRESHOLD:
                    casual_descriptions.append(f"many {label}s")
                else:
                    casual_descriptions.append(f"{count} {label}" + ("s" if count > 1 else ""))
            self.last_casual_spoken = now
            speech_lines.append("There are " + " and ".join(casual_descriptions) + " around.")

        return speech_lines


# TODO: add after history = detection.DetectionHistory()
speech_controller = speech_logic.SpeechController()

# TODO: update speaker_worker()
def speaker_worker():
    """Constantly speaks out loud filtered and prioritized detections"""
    while True:
        detections = detection_queue.get()
        if detections is None:
            break

        messages = speech_controller.summarize_detections(detections)
        for msg in messages:
            print(f"[TTS] {msg}")
            speak(msg)

# TODO: Tune Detectionhistory and Queue logic
if current_detections != [] and history.has_changed(current_detections):
    detection_queue.put(current_detections)


CUSTOM_PHRASES = {
    "crosswalks": "Crosswalks detected {depth:.1f} meters ahead.",
    "stairs": "Stairs ahead at {depth:.1f} meters. Please be careful.",
    "bus": "A bus is nearby at {depth:.1f} meters.",
}

msg = CUSTOM_PHRASES[label].format(depth=d.depth)
