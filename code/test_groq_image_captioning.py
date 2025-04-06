import cv2
import io
import torch
import depthai as dai
import base64
import requests
import time
import json


# === Groq API Setup ===
# Getting API key (you will need to generate one for yourself)
#with open('../auxiliary/config.json') as file:
with open('../auxiliary/config_secret.json') as file:
    data = json.load(file)
    GROQ_API_KEY = data['GROQ_API_KEY']

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}


# === Util: Encode an image as base64 JPEG ===
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# === Util: Encode OpenCV image as base64 JPEG ===
def encode_image_from_cv2(image):
    _, buffer = cv2.imencode(".jpg", image) 
    return base64.b64encode(buffer).decode("utf-8")


# === Util: Send a request to groq ===
def query_groq_with_image(base64_image):
    # The image to be sent to the model (base64 format, received in function's arguments)
    image_content = {
        "type": "image_url", 
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
    
    # The prompt to be sent alongside the image (what to do with it)
    prompt = "In a short sentence, please describe the image to a blind person."
    
    # Data of the request, includes the model selected for the task
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ]
            }
        ]
    }

    try:
        response = requests.post(GROQ_URL, headers=GROQ_HEADERS, json=data)
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            print(f"\n🧠 AI description: {text}\n")
        else:
            print(f"⚠️ Error {response.status_code}: {response.text}")
    except Exception as e:
        print("❌ Exception:", e)


# === Device Check ===
print('cv2 cuda enabled:', bool(cv2.cuda.getCudaEnabledDeviceCount()))
print('torch using device:', torch.cuda.get_device_name(0))
print('depthai using device:', dai.Device.getAllAvailableDevices())

# === Pipeline Setup ===
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# === Runtime Variables ===
capture_interval = 60  # seconds
first_interval = capture_interval-10 # seconds
last_capture_time = time.time() - first_interval  # makes sure the first image is captured after ~10sec

# === Main Loop ===
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=8, blocking=False)

    while True:
        frame = q_video.get().getCvFrame()

        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Resize with GPU
        gpu_frame = cv2.cuda.resize(gpu_frame, (128, 128))

        # Download to CPU for display and encoding
        frame_resized = gpu_frame.download()

        # Show preview
        cv2.imshow("OAK-D Lite Preview (CUDA)", frame_resized)

        # Every 60 seconds: encode & send to Groq
        current_time = time.time()
        if current_time - last_capture_time > capture_interval:
            last_capture_time = current_time

            # Convert to base64
            b64_image = encode_image_from_cv2(frame_resized)

            # Query Groq with the image prompt
            query_groq_with_image(b64_image)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
