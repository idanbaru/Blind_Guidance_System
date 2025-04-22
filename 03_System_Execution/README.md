# System Execution

This directory contains the implementation and execution files for the Blind Guidance System. Below is an overview of the files and their purpose.

## Files Overview

- **`main.py`**: The main script to execute the Blind Guidance System. It integrates all threads and handles the system's workflow.
- **`detection.py`**: Detection handaling relevant code.
- **`oakd_configuration.py`**: Configuration file for the OAK-D Lite camera, including settings for video capture and executing Yolov8.
- **`snapshot_captioning.py`**: Code for handling environment description using Groq API and Llama 4 model.
- **`utils.py`**: Utility functions used across the system for common tasks.
- **`README.md`**: This documentation file.

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
