import cv2
import torch
import depthai as dai
import tensorrt as trt
import numpy as np

# TensorRT Inference
def load_trt_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def inference_trt(engine, image):
    # Prepare buffers for input and output
    context = engine.create_execution_context()
    input_shape = (1, 3, 640, 640)  # Assuming YOLOv8n model with 640x640 input

    # Allocate memory for input and output buffers
    input_buffer = np.ascontiguousarray(image)
    input_buffer = np.transpose(input_buffer, (2, 0, 1))  # HWC to CHW
    input_buffer = np.expand_dims(input_buffer, axis=0)  # Batch size of 1

    output_shape = (1, 25200, 85)  # Assuming YOLO output shape (adjust for YOLO11n if needed)
    output_buffer = np.zeros(output_shape, dtype=np.float32)

    # Run inference
    bindings = [int(input_buffer.ctypes.data), int(output_buffer.ctypes.data)]
    context.execute_v2(bindings)

    # Post-process the output (e.g., Non-Maximum Suppression, etc.)
    return output_buffer

# DepthAI Pipeline Setup
print('cv2 cuda enabled:', bool(cv2.cuda.getCudaEnabledDeviceCount()))
print('torch using device:', torch.cuda.get_device_name(0))
print('depthai using device:', dai.Device.getAllAvailableDevices())

# Create a DepthAI pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Load TensorRT model
engine_path = "/home/jetson/Blind_Guidance_System/code/models/yolov8n.engine"  # Path to your exported TensorRT model
engine = load_trt_engine(engine_path)

# Connect to DepthAI device and start pipeline
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=8, blocking=False)
    
    while True:
        frame = q_video.get().getCvFrame()

        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Resize frame using GPU
        gpu_frame = cv2.cuda.resize(gpu_frame, (640, 640)) 

        # Download back to CPU for processing
        frame = gpu_frame.download()

        # Run TensorRT inference
        output = inference_trt(engine, frame)

        # Process output (e.g., NMS, bounding box drawing)
        for detection in output[0]:
            confidence = detection[4]
            if confidence > 0.5:  # Filter out low-confidence detections
                x1, y1, x2, y2 = detection[:4]  # Bounding box coordinates
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = "Object"  # Replace with actual class names if needed
                cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("OAK-D Lite Preview (CUDA)", frame)

        # Check if the 'q' key was pressed to stop the process
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
