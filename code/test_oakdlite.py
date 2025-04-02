import cv2
import torch
import depthai as dai
print('cv2 cuda enabled:' ,bool(cv2.cuda.getCudaEnabledDeviceCount()))
print('torch using device:', torch.cuda.get_device_name(0))
print('depthai using device: ', dai.Device.getAllAvailableDevices())

### WITH GPU SUPPORT
# Create a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)  # Ensure it's in BGR format


# Create an output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=8, blocking=False)
    
    while True:
        frame = q_video.get().getCvFrame()

        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Convert color (not needed - cv2 works in BGR as well)
        #gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)

        # Resize using GPU
        gpu_frame = cv2.cuda.resize(gpu_frame, (640, 640)) 

        # Download back to CPU only for display
        frame = gpu_frame.download()

        # Display optimized frame
        cv2.imshow("OAK-D Lite Preview (CUDA)", frame)

        # Check (every millisecond) if the 'q' key was press to stop the process
        if cv2.waitKey(1) == ord('q'):
            break

        # Check if window is closed
        #if cv2.getWindowProperty("OAK-D Lite Preview (CUDA)", cv2.WND_PROP_VISIBLE) < 1:
        #    break

cv2.destroyAllWindows()
