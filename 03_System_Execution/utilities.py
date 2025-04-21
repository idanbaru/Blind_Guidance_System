import socket
import threading

# ===== DEFINITIONS =====
TTS_COOLDOWN = 2 # seconds
ONLINE_CHECK_TIMEOUT = 2 # seconds
OFFLINE_TRANSITION_TEXT = "System now working offline..."
ONLINE_TRANSITION_TEXT = "System back online!"

class RingBufferQueue:
    def __init__(self, maxsize):
        self.buffer = [None] * maxsize
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.write_index = 0
        self.read_index = 0
        self.count = 0

    def put(self, item):
        with self.lock:
            self.buffer[self.write_index] = item
            self.write_index = (self.write_index + 1) % self.maxsize
            if self.count < self.maxsize:
                self.count += 1
            else:
                # Overwrite oldest -> move read pointer forward
                self.read_index = (self.read_index + 1) % self.maxsize

    def get(self):
        while True:
            with self.lock:
                if self.count == 0:
                    continue  # busy wait (can also add `time.sleep(0.01)` to reduce CPU)
                item = self.buffer[self.read_index]
                self.read_index = (self.read_index + 1) % self.maxsize
                self.count -= 1
                return item

    def clear(self):
        with self.lock:
            self.write_index = 0
            self.read_index = 0
            self.count = 0

def is_online(host="8.8.8.8", port=53, timeout=ONLINE_CHECK_TIMEOUT):
    # Tries to form a TCP connection to Google's DNS for 200ms max
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except socket.error:
        return False


