
import os
import torch
torch.cuda.init()

from ultralytics import YOLO

# 1. Load the model used in your code
# Replace 'yolov8s.pt' with the exact path to your trained model if different.
model = YOLO('yolov8s.pt') 

# 2. Export it to 'engine' format (TensorRT)
# The 'half' argument enables FP16 precision for maximum speed on Jetson.
# The 'device=0' argument explicitly uses the first GPU.
success = model.export(
    format='engine',
    half=True,  # Uses FP16 precision
    device=0    # Uses GPU 0
) 

if success:
    print(f"\n✅ TensorRT engine successfully created: {success}")
    print("You can now use this file in your ObjectDetector.")
else:
    print("\n❌ Failed to create the TensorRT engine.")
