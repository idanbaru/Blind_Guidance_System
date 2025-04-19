import os
import sys
import yaml
import json
import torch_directml
import shutil
import xmltodict
import numpy as np
import pandas as pd
import ultralytics
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

print("Torch DirectML:", torch_directml.is_available())
print("GPU name:", torch_directml.device_name(0))

input_dir = 'annotated_outdoor_yolo'
output_dir = ''
temp_dir = 'temp/'

images_dir = os.path.join(input_dir, 'images')
labels_dir = os.path.join(input_dir, 'labels')

# Get the list of image files (assuming all formats are .jpg, change if needed)
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

# Create dataset indices
dataset_size = len(image_files)
indices = list(range(dataset_size))

# Split into train (70%), val (10%), test (20%)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=42)

# Store indices in a dictionary
split_data = {
    "train": train_indices,
    "val": val_indices,
    "test": test_indices
}

print(f"Train set size: {len(train_indices)} - {len(train_indices)/len(indices)*100:.2f}%")
print(f"Validation set size: {len(val_indices)} - {len(val_indices)/len(indices)*100:.2f}%")
print(f"Test set size: {len(test_indices)} - {len(test_indices)/len(indices)*100:.2f}%\n")

# Save the split indices as JSON
json_split_path = os.path.join(output_dir, "dataset_split.json")
with open(json_split_path, "w") as f:
    json.dump(split_data, f, indent=4)

print(f"Dataset split saved to {json_split_path}")

