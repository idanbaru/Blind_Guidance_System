check if trt is installed
    dpkg -l | grep nvinfer

install trt (if not installed)
    sudo apt update
    sudo apt install -y nvidia-tensorrt


hotfix, if 'trtexec' command not found:
    sudo chmod +x /usr/src/tensorrt/bin/trtexec
    sudo ln -s /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec

now convert from onnx (get onnx from ultralytics container)
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16

===========================================================================

Using pt2trt
    sudo apt update
    sudo apt install -y python3-packaging


    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo python3 setup.py install
    
     cp ~/Blind_Guidance_System/code/models/yolov8n.onnx ./

Now in python file execute (from torch2trt folder)
    from torch2trt import torch2trt
    import torch

    model_trt = torch2trt(model, [torch.randn(1, 3, 640, 640).cuda()])
    torch.save(model_trt.state_dict(), "yolov8n_trt.pth")
