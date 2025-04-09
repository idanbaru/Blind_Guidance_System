
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
        return (self.label == other.label and
                abs(self.center[0] - other.center[0]) < 25 and
                abs(self.center[1] - other.center[1]) < 25 and
                abs(self.depth - other.depth) < 2) # allow small fluctuations (TODO: edit these magic numbers)

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
    
