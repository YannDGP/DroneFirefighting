
# FDAR – Fire Detection and Ranging

A modular vision-based system for real-time fire, smoke, and human detection with 3D localization, designed for autonomous firefighting drones. The pipeline integrates RGB and thermal sensing, object detection, monocular depth estimation, and safety-aware targeting to support autonomous fire suppression.

## Features

- **Multi-Modal Perception**
  - Supports RGB-only, thermal-only.
- **Real-Time Detection**
  - Includes YOLO (v5–v12) and transformer-based detectors (RT-DETRv2, RF-DETR).
  - Multi-class detection of fire, smoke, and humans.
- **Depth Estimation**
  - Monocular depth prediction using Depth Anything V2 with ZoeDepth scaling.
  - 3D localization of all detections.
- **Fire Targeting & Safety**
  - Lightweight fire source localization inside bounding boxes.
  - Human proximity checks with `safe-to-suppress` flags.
  - Optional pseudo-3D bounding box visualization.

## Performance
Tested on **NVIDIA Jetson Orin Nano** (JetPack 6, CUDA 12.2), live camera stream via ROS2 Humble.

| Configuration | Inference Speed |
|---|---|
| Original sequential pipeline | ~2 FPS |
| Multi-threaded + TensorRT (FP16) | ~20 FPS (10–25 FPS range) |

- Fire detection confidence threshold: 0.3–0.4 for clear flames (detections below 0.1 discarded)
- Depth estimation error: ±0.25m at 3m distance
- Reliable ranging range after polynomial correction: 3m – 5m
- ROS2 camera Workswell WIRIS Enterprise latency: ~500–700ms (reduced from >1s via resolution optimization)
- No latency on ZED2 Stereolabs

## Requirements

- Python 3.8+
- PyTorch and TochVision Unique Build 
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

To install the wheels of Pytorch and TorchVision
```bash
#Once you clone the repository, use 
git lfs pull 
#The wheels are too heavy for Git
```

## Project in Jetson Orin Nano

Everything was tested and made on the Jetson Orin Nano
The project is already inside (Last Test 24/02/2026)
To launch the project :

First Connect the camera model Workswell Wiris Enterprise
```bash
source /opt/ros/humble/setup.bash
#Inside the ros_ws project folder
source install/setup.bash
ros2 run workswell_ros2_interface camera_node
```
In another terminal, to run the FDAR script:
```bash
cd Documents/Stage_Saxion
#For the protection of the project I used a venv
#Launch the venv
source venv venv_saxion_project/bin/bash
#Once the venv Launch
cd DroneFirefighting
#The Main folder project of my work on the Jetson
source /opt/ros/humble/setup.bash
python Ros_run.py
```

## Download of the project (Made for the Jetson)
```bash
git clone <>
cd <folder_name>
pip install -r requirements.txt
pip install torch-2.3.0a0+git97ff6cf-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

## Installation on Nvidia Jetson Orin Nano

Install ROS2 on the Jetson Orin Nano (Humble).
Connect the RGB–thermal camera to the Jetson.


Start the camera through ROS2 (With the Workswell Camera):

```bash
source /opt/ros/humble/setup.bash
#Inside the ros project folder
source install/setup.bash
ros2 run workswell_ros2_interface camera_node
```
In another terminal, source ROS2 and run the FDAR script:
```bash
source /opt/ros/humble/setup.bash
python Ros_run.py
```

### Configuration Options

You can modify the following parameters in `Ros_run.py`:

- **Input**:
  - `CameraNode` : In the subscriber you can change the topic (for image and focal)

- **Model Settings**:
  - `yolo_model`: YOLOv5–YOLOv12, RT-DETRv2, or RF-DETR
  - `depth_model`:  Depth Anything v2 model size: "small", "base", "large"    `OR`   Zoedepth ['train', 'infer', 'eval']

- **Detection Settings**:
  - `conf_threshold`: Confidence threshold for object detection
  - `iou_threshold`: IoU threshold for NMS
  - `classes`: Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes

- **Feature Toggles**:
  - `TensorBoard`: For post video analysis

## Project Structure

```
FDAR/
│── Ros_run.py                # Main entry point
│── detection_model.py    # Multi-modal object detection logic
│── depth_model.py        # Monocular depth estimation + scaling
│── bbox3d_utils.py       # 3D bounding box and projection utilities
│── load_camera_params.py # Camera intrinsics & extrinsics loader
├── requirements.txt      # Dependencies
└── README.md             # This file


## How It Works

1. **Object Detection**: Detects fire, smoke, and humans from RGB, thermal, or fused RGT images.
2. **Depth Estimation**: Generates relative depth maps and scales them to metric depth.

