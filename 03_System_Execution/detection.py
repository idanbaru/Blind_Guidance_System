import time
import oakd_configuration
from collections import defaultdict


URGENT_CLASSES = {"crosswalk", "stairs assending", "stairs descending", "bus"}
CASUAL_CLASSES_OUTDOOR = {"person", "car", "bus station"}
CASUAL_CLASSES_INDOOR = set(oakd_configuration.labelMap_indoor_config)
CASUAL_CLASSES = CASUAL_CLASSES_OUTDOOR | CASUAL_CLASSES_INDOOR
MANY_DETECTION_THRESHOLD = 3
CASUAL_THRESHOLD = 5
CASUAL_TIMEOUT = 10  # seconds before repeating casual summary
URGENT_TIMEOUT = 5  # seconds before repeating urgent warning


class SpeechController:
    def __init__(self):
        self.last_urgent_spoken = {}  # label -> timestamp
        self.last_casual_spoken = 0  # timestamp

    def summarize_detections(self, detections):
        """Return a spoken string based on detection priority and frequency"""
        now = time.time()
        urgent_lines = []
        casual_detections = defaultdict(lambda: {"count": 0, "depth": [], "location": []})

        # if there are not detections return None
        if len(detections) == 0:
            return None
        
        for d in detections:
            if d.label in URGENT_CLASSES:
                if (now - self.last_urgent_spoken.get(d.label, 0)) > URGENT_TIMEOUT:
                    urgent_lines.append(f"{d.label} ahead at {d.depth:.1f} meters on your {d.location}")
                    self.last_urgent_spoken[d.label] = now
            elif d.label in CASUAL_CLASSES:
                casual_detections[d.label]["count"] += 1
                casual_detections[d.label]["depth"].append(d.depth)
                casual_detections[d.label]["location"].append(d.location)

        # Urgent always gets priority
        if urgent_lines:
            return "Attention: " + ", ".join(urgent_lines)
        
        # Casual Classes:
        # Add casual if timeout has passed and not already speaking urgent
        if not urgent_lines and (now - self.last_casual_spoken) > CASUAL_TIMEOUT and len(casual_detections) > 0:
            # if there are <= 3 detections also say the depth
            say_depth = True if (sum(1 for d in detections) <= MANY_DETECTION_THRESHOLD) else False
            casual_descriptions = []
            single_detection = False
            for label, info in casual_detections.items():                  
                count = info["count"]
                depth = info["depth"]
                location = info["location"]
                
                if label == "person" and count > MANY_DETECTION_THRESHOLD:
                    label = "people"
                if count > CASUAL_THRESHOLD:
                    casual_descriptions.append(f"many {label}s")
                else:
                    if count == 1:
                        single_detection = True
                        if say_depth:
                            casual_descriptions.append(f"a {label} at {depth[0]:.1f} meters on your {location[0]}")
                        else:
                            casual_descriptions.append(f"a {label}")
                    else:
                        if say_depth:
                            for i in range(count):
                                casual_descriptions.append(f"a {label} at {depth[i]:.1f} meters on your {location[i]}")
                        else:
                            casual_descriptions.append(f"{count} {label}" + ("s" if (count > 1 and label not in ["people"]) else ""))
            self.last_casual_spoken = now
            if single_detection and len(casual_descriptions) == 1:
                return f"There is {casual_descriptions[0]} " + ("around" if (not say_depth) else "")
            else:
                return f"There are " + " and ".join(casual_descriptions) + ("around" if (not say_depth) else "")

class Detection:
    def __init__(self, label, center=(0,0), depth=0):
        self.label = label
        self.center = center
        self.depth = depth
        self.location = self.direction()
    
    # TODO: edit magic 25 and 2 numbers depends on image resulotion
    def __eq__(self, other):
        if not isinstance(other, Detection):
            return False
        return self.label == other.label # TODO: implement the more sophisticated logic
        #return (self.label == other.label and
        #        abs(self.center[0] - other.center[0]) < 25 and
        #        abs(self.center[1] - other.center[1]) < 25 and
        #        abs(self.depth - other.depth) < 2) # allow small fluctuations (TODO: edit these magic numbers)

    def direction(self):
        # TODO: define relative img size
        IMG_SIZE = 640
        if self.center[0] < IMG_SIZE / 3:
            return "left"
        elif self.center[0] > 2*IMG_SIZE / 3:
            return "right"
        else:
            return "middle"

    # TODO: edit different cases for different labels, for example:
    # if label in requires_attention return Attention! {self.label} coming from the {direction(self.center[0])} in {self.depth} meters.
    def __repr__(self):
        return f"{self.label} in {self.depth:.2f} meters to the {self.direction()}"
