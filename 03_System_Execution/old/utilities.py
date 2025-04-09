import socket

# ===== DEFINITIONS =====
TTS_COOLDOWN = 2 # seconds
ONLINE_CHECK_TIMEOUT = 0.2 # seconds
OFFLINE_TRANSITION_TEXT = "System now working offline..."
ONLINE_TRANSITION_TEXT = "System back online!"


def is_online(host="8.8.8.8", port=53, timeout=ONLINE_CHECK_TIMEOUT):
    # Tries to form a TCP connection to Google's DNS for 200ms max
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except socket.error:
        return False


