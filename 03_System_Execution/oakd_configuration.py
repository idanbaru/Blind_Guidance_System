import depthai as dai
from pathlib import Path

IMG_SIZE = 640
DEFAULT_FPS = 30
IOU_THRESHOLD = 0.5 # RECOMMENDED
CONFIDENCE_THRESHOLD = 0.5

BB_SCALE_FACTOR = 0.5
DEPTH_THRESHOLD_LOW = 100 # in millimeters
DEPTH_THRESHOLD_HIGH = 100000 # in millimeters

labelMap_outdoor = [
            "bus",
            "bus_station",
            "car",
            "crosswalk",
            "person",
            "pothole",
            "stairs_down",
            "stairs_up"
        ]
labelMap_indoor = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ]

labelMap_indoor_config = [
        "person", "dog",         "cat",        "backpack",     "umbrella",   "suitcase",
        "bottle", "wine glass",  "cup",        "fork",         "knife",      "spoon",
        "bowl",   "banana",      "apple",      "sandwich",     "orange",     "broccoli",
        "carrot", "hot dog",     "pizza",      "donut",        "cake",       "chair",
        "sofa",   "pottedplant", "bed",        "diningtable",  "toilet",     "tvmonitor",
        "laptop", "mouse",       "remote",     "keyboard",     "cell phone", "microwave", 
        "oven",   "toaster",     "sink",       "refrigerator", "book",       "clock",
        "vase",   "scissors",    "hair drier", "toothbrush"
]

def get_nn_path():
    # Get yolo model blob file path in an absolute format
    #TODO: change the model path depending on a flag
    nnPath = str((Path(__file__).parent.parent / Path('auxiliary/models/yolov8n_indoor_5shave.blob')).resolve().absolute())
    print(f"Importing neural network from: {nnPath}.")
    if not Path(nnPath).exists():
        raise FileNotFoundError(f'Required file/s not found')
    return nnPath

def get_label_map(mode="indoor"):
    if mode == "indoor":
        return labelMap_indoor
    elif mode == "outdoor":
        return labelMap_outdoor
    else:
        raise ValueError("Invalid mode. Please specify either 'indoor' or 'outdoor'.")

def configure_oakd_camera(syncNN=True, nnPath="", mode="indoor"):
    # Set neural network path to default in case a path was not passed as an argument
    if nnPath == "":
        nnPath = get_nn_path()

    labelMap = get_label_map(mode=mode)
    
    # Create pipeline and define sources and outputs
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    #detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    # Add depth pipeline (TODO: check this)
    depth = pipeline.create(dai.node.StereoDepth)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Define properties
    camRgb.setPreviewSize(IMG_SIZE, IMG_SIZE)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(DEFAULT_FPS)

    # Configure Mono Cameras (TODO: check this)
    for monoCam, socket in [(monoLeft, dai.CameraBoardSocket.LEFT), (monoRight, dai.CameraBoardSocket.RIGHT)]:
        monoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoCam.setBoardSocket(socket)

    # Configure StereoDepth
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB
    depth.setSubpixel(True)  # Optional: improves accuracy

    # Optional: tuning near/far range
    #depth.setDepthLowerThreshold(100)   # mm
    #depth.setDepthUpperThreshold(10000) # mm

    # Define the neural network's settings
    detectionNetwork.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detectionNetwork.setNumClasses(len(labelMap))
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setIouThreshold(IOU_THRESHOLD)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.input.setBlocking(False)

    # Settings for the Stereo to neural network connection (TODO: check)
    # detectionNetwork.setBoundingBoxScaleFactor(BB_SCALE_FACTOR)  # shrink box for depth avg
    detectionNetwork.setDepthLowerThreshold(DEPTH_THRESHOLD_LOW)
    detectionNetwork.setDepthUpperThreshold(DEPTH_THRESHOLD_HIGH)

    # Enable spatial detection and sets the depth of each detection to be the average of points
    #detectionNetwork.setSpatialBoundingBoxScaleFactor(0.5)
    detectionNetwork.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.MIN)


    # Linking (syncNN=True to pass video stream through the neurtal network) 
    camRgb.preview.link(detectionNetwork.input)
    
    if syncNN:
        detectionNetwork.passthrough.link(xoutRgb.input)
        #camRgb.video.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    # Connect depth to the neural network (TODO: check)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(detectionNetwork.inputDepth)

    return pipeline


if __name__ == "__main__":
    import cv2
    import numpy as np
    import time

    # Check if the script is running correctly on indoor mode
    
    syncNN = True
    
    pipeline = configure_oakd_camera(syncNN=syncNN)
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        color2 = (255, 255, 255)

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def displayFrame(name, frame):
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap_indoor[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Show the frame
            if video_writer_detection is not None:
                video_writer_detection.write(frame)
            cv2.imshow(name, frame)

        # Parameters for video saving
        output_filename = "output.avi"        # Output file name for raw video
        detection_filename = "detection.avi"  # Output file name for video with detections
        frame_width = 640                     # Width of the video frame
        frame_height = 480                    # Height of the video frame
        fps = 30                              # Frames per second
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID for .avi format)
        
        # Initialize the VideoWriter for raw video
        video_writer_raw = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        # Variables to handle detection video saving
        video_writer_detection = None  # Will be initialized after the first detection

        # Initialize counter and startTime
        counter = 0
        startTime = time.monotonic()
        
        while True:
            if syncNN:
                inRgb = qRgb.get()
                inDet = qDet.get()
            else:
                inRgb = qRgb.tryGet()
                inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

            if inDet is not None:
                detections = inDet.detections
                counter += 1
                
                # Initialize video writer for detection if it hasn't been initialized
                if video_writer_detection is None:
                    video_writer_detection = cv2.VideoWriter(detection_filename, fourcc, fps, (frame_width, frame_height))

            if frame is not None:
                video_writer_raw.write(frame)
                displayFrame("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
            
    video_writer_raw.release()
    if video_writer_detection is not None:
        video_writer_detection.release()
    cv2.destroyAllWindows()
