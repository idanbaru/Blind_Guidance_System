from ultralytics import YOLO

def download_and_export(model_name):
    print(f"Downloading {model_name}...")
    model = YOLO(model_name)
    print(f"Exporting {model_name} to TensorRT...")
    model.export(format='engine')
    print(f"{model_name} exported successfully!")

if __name__ == "__main__":
    models = ["yolov8n", "yolo11n"]
    for model in models:
        download_and_export(model)
