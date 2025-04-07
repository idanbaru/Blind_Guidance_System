## System Design
# Main Thread - Runs OAK-D pipeline, handles YOLO detections (offline)
# Speaker Thread - Speaks new/changed detections from a queue (online/offline)
# Describer Thread - Describes the environment every 60 seconds (online)

