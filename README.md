# <h1 align="center">ECE 00440169 - Technion - CRML - Project B </h1> 
<h1 align="center">Blind Guidance System</h1>
<p align="center"><strong>An AI Powered System for the Visually Impaired</strong></p>

<h4 align="center">
  <table align="center" style="border: none;">
    <tr style="border: none;">
      <td align="center" style="border: none;">
        <div>
          <img src="./auxiliary/readme/Itai.png" width="100" height="100"/> <br>
          <strong>Itai Benyamin</strong> <br>
          <a href="https://www.linkedin.com/in/itai-benyamin/">
            <img src="./auxiliary/readme/LinkedInLogo.png" width="40" height="40"/>
          </a>
          <a href="https://github.com/Itai-b">
            <img src="./auxiliary/readme/GitHubLogo.png" width="40" height="40"/>
          </a>
        </div>
      </td>
      <td align="center" style="border: none;">
        <div>
          <img src="./auxiliary/readme/Idan.png" width="100" height="100"/> <br>
          <strong>Idan Baruch</strong> <br>
          <a href="https://www.linkedin.com/in/idan-baruch-76490a181/">
            <img src="./auxiliary/readme/LinkedInLogo.png" width="40" height="40"/>
          </a>
          <a href="https://github.com/idanbaru">
            <img src="./auxiliary/readme/GitHubLogo.png" width="40" height="40"/>
          </a>
        </div>
      </td>
    </tr>
    <tr style="border: none;">
      <td colspan="2" align="center" style="border: none">
        <a href="https://youtu.be/Lghsg8BiNpw" target="_blank">
          <img src="./auxiliary/readme/YouTubeLogo.png" width="50" height="50"/>
        </a>
      </td>
    </tr>
  </table>
</h4>

## Abstract
This project aims to develop a mountable system that processes the video stream from a camera and translates it into audio output in real time. The system consists of a portable processor with a GPU (Jetson Nano) and an AI-enabled camera (OAK-D Lite). It uses lightweight models such as Yolov8n to perform visual tasks such as object detection and depth estimation, differentiating between indoor and outdoor modes, each focusing on relevant object types for a visually impaired individual. The system supports both online and offline text-to-speech conversion, seeking to reduce computation load while also providing stability for internet instabilities. The results are communicated through spoken descriptions or audio outputs. The system delivers timely, meaningful feedback while maintaining portability, reliability, and power efficiency.
<div align="center">
  <img src="./auxiliary/readme/System.jpg" alt="System" width="400">
</div>

## Repository Structure

| File Name                       | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| `01_Setup`               | Instructions and Scripts on how to setup your own system to run the code   |
| `02_Model_Training`                    | Files for training the outdoor Yolov8 model|
| `03_System_Execution`                  | The Code|

Each folder contains a README file with more details on the files and their purpose.

## How to Run the Code
After setting up the system (see `01_Setup`), you can run the code by executing the following command in the terminal:
```bash
python3 03_System_Execution/main.py {--mode <mode>} {--record} 
```
where:
- `--help`: if specified, the system will print the help message and exit.
- `--mode <mode>`: the mode of the system. The options are:
  - `outdoor`: for outdoor mode
  - `indoor`: for indoor mode
  if not specified, the system will run in indoor mode.
- `--record`: If specified, the system records camera footage in 30-second .mp4 files, saved under `03_System_Execution/recorded_videos/recorded_videos_<timestamp>`, where `<timestamp>` is the run’s epoch time. Each file is named `video_<timestamp>.mp4`, with a new file created every 30 seconds.

## System's Overview

### System's Components

- **Nvidia Jetson Nano Developer Kit** – a compact and energy-efficient micropro-
cessor, equipped with a graphics processing unit (GPU) designed for computer vision
and AI computations.
- **Luxonis OAK-D Lite** - an affordable AI-enabled depth camera that integrates
stereo vision, RGB imaging, and an on-device neural inference engine for real-time
spatial AI applications.
Implementation 11
#### Mandatory Accessories

- **MicroSD Card** - main storage of the Jetson Nano Developer Kit, recommended
at least 64 GB UHS-1.
- **USB Wi-Fi Adapter** - allows accessing online TTS and image captioning models.
- **Small Speaker** - voices the alerts and information processed by the system.
- **Portable Battery or Power Bank** - powers the portable system.

#### Optional Accessories
- **Jetson Nano Metal Case** - provides protection and simplifies fixing the camera on
the Jetson Nano as well as mounting the entire system on a GDR.
- **5 V PWM Fan** - allows custom control on system’s heat.

### System's Architecture
<div align="center">
  <img src="./auxiliary/readme/Architecture.png" alt="Architecture" width="600">
</div>
In this project’s design, AI-enabled cameras are used, which are capable of not only
capturing a scene but also running basic AI models of object detection. Both the video stream
as well as information about the detections in the current captured frame are then transferred
from the camera and vision unit (i.e., the AI-enabled camera) to the main processing unit,
which controls the entire system. The processing unit constantly reads the incoming frames
and respective detections from the camera and vision unit. These are stored at a designated
buffer, which the processing unit sorts and prioritizes. Afterwards, it constructs alerts and
sentences as text, and then runs a TTS model to synthesize the audio output. The processing
unit is also capable of handling requests sent to external cloud services to provide access to
stronger, more enhanced models.

## System's Algorithm
<div align="center">
  <img src="./auxiliary/readme/Algorithm.png" alt="Algorithm" width="600">
</div>

## Future Work
<div align="center">
  <img src="./auxiliary/readme/FutureWork.png" alt="FutureWork" width="600">
</div>

The system can be further improved in several ways:

1. **Online-Only System**	
  - Leverage cloud-based models for improved functionality and model complexity (use of different SOTA models)
  - Simplify hardware requirements and reduce power consumption	
  Potential for real-time updates and continuous model improvement

2. Offline-Only System	
  - Fully self-contained, privacy-preserving solution
  - Ideal for low-connectivity environments
  - Requires hardware upgrades and additional power draw

3. Navigation Capabilities	
  - Add GPS integration and route guidance for outdoor navigation



