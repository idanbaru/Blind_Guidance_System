import depthai as dai
from pathlib import Path

IMG_SIZE = 640
DEFAULT_FPS = 30
IOU_THRESHOLD = 0.5 # RECOMMENDED
CONFIDENCE_THRESHOLD = 0.5

BB_SCALE_FACTOR = 0.5
DEPTH_THRESHOLD_LOW = 100 # in millimeters
DEPTH_THRESHOLD_HIGH = 10000 # in millimeters


labelMap = [
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

def get_nn_path():
    # Get yolo model blob file path in an absolute format
    nnPath = str((Path(__file__).parent.parent / Path('auxiliary/models/yolov8n_openvino_2022.1_6shave.blob')).resolve().absolute())
    #nnPath = str((Path(__file__).parent / Path('yolo11n.blob')).resolve().absolute()) # OAK-D LITE SUPPORTS YOLOv5~v8 (!!!)
    print(f"Importing neural network from: {nnPath}.")
    if not Path(nnPath).exists():
        raise FileNotFoundError(f'Required file/s not found')
    return nnPath


def configure_oakd_camera(syncNN=True, nnPath=""):
    # Set neural network path to default in case a path was not passed as an argument
    if nnPath == "":
        nnPath = get_nn_path()

    # Create pipeline and define sources and outputs
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    #detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)
    
    # TODO: CHECK THIS!!!
    #spatialDataOut = pipeline.create(dai.node.XLinkOut)
    #spatialDataOut.setStreamName("spatial")
    #detectionNetwork.spatialLocations.link(spatialDataOut.input)

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
    detectionNetwork.setBoundingBoxScaleFactor(BB_SCALE_FACTOR)  # shrink box for depth avg
    detectionNetwork.setDepthLowerThreshold(DEPTH_THRESHOLD_LOW)
    detectionNetwork.setDepthUpperThreshold(DEPTH_THRESHOLD_HIGH)

    # Enable spatial detection and sets the depth of each detection to be the average of points
    #detectionNetwork.setSpatialBoundingBoxScaleFactor(0.5)
    detectionNetwork.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.AVERAGE)


    # Linking (syncNN=True to pass video stream through the neurtal network) 
    camRgb.preview.link(detectionNetwork.input)
    
    if syncNN:
        #detectionNetwork.passthrough.link(xoutRgb.input)
        camRgb.video.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    # Connect depth to the neural network (TODO: check)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(detectionNetwork.inputDepth)

    return pipeline

