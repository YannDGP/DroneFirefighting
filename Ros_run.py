import cv2
import torch
import time
import numpy as np
from threading import Thread, Lock
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from torch.utils.tensorboard import SummaryWriter
import datetime
import signal
import sys

# --- Imports ROS 2 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data

# --- Limits of frames ---
TOLERANCE_DETECTION = 3
TOLERANCE_DEPTH = 5

# --- Unique Class for the Camera gestion (Info + Images) ---
class CameraNode(Node):
    def __init__(self, frame_source, lock):
        super().__init__('camera_node')
        self.frame_source = frame_source
        self.lock = lock
        self.fx = None # stock focal here

        # 1. Subscriber for the Images (RGB)
        self.image_sub = self.create_subscription(
            Image,
            '/workswell/rgb',
            self.image_callback,
            qos_profile=qos_profile_sensor_data)

''' Using the Polynomial corrector in Depth, the focal is not needed

        # 2. Subscriber for the CameraInfo (Focale)
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/x500/camera/camera_info',
            self.info_callback,
            10) # QoS by default
'''
    def info_callback(self, msg):
        # We get the focal once
        if self.fx is None:
            self.fx = msg.k[0]
            self.get_logger().info(f"Focal received : {self.fx}")

    def image_callback(self, msg):
        # Somme cameras send images in BGRA (4 canals), preventive reshape
        image_data = np.frombuffer(msg.data, dtype=np.uint8)

        try:
            # Dynamic gestion of canals if it changes (3 or 4)
            nb_channels = int(image_data.size / (msg.height * msg.width))
            frame = image_data.reshape((msg.height, msg.width, nb_channels))

            # If 4 canals (BGRA), convertion in BGR for OpenCV/Models
            if nb_channels == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        except ValueError as e:
            self.get_logger().error(f"Reshape Error: {e}.")
            return

        with self.lock:
            self.frame_source["latest_frame"] = frame.copy()
            self.frame_source["frame_id"] += 1

# --- Workers ---
def detection_worker(detector, frame_source, output_dict, lock, stop_flag):
    last_processed_id = -1
    while not stop_flag["stop"]:
        frame = None
        current_id = -1
        current_depth_map = None

        with lock:
            if "latest_frame" in frame_source and frame_source["frame_id"] > last_processed_id:
                frame = frame_source["latest_frame"].copy()
                current_id = frame_source["frame_id"]
            current_depth_map = output_dict.get("raw_depth_map",None)

        if frame is not None:
            start_detect = time.time()
            small_frame = cv2.resize(frame, (640, 360))
            detection_frame, detection_data, _ = detector.detect(small_frame, track=False, depth_map=current_depth_map)
            end_detect = time.time()
            detect_fps = 1 / (end_detect - start_detect) if (end_detect - start_detect) > 0 else 0

            with lock:
                output_dict["detect"] = detection_frame
                output_dict["detect_id"] = current_id
                output_dict["detect_fps"] = detect_fps
                output_dict["latest_detections"] = detection_data

            last_processed_id = current_id
        time.sleep(0.001)

def depth_worker(depth_estimator, frame_source, output_dict, lock, stop_flag):
    last_processed_id = -1
    while not stop_flag["stop"]:
        frame = None
        current_id = -1

        with lock:
            if "latest_frame" in frame_source and frame_source["frame_id"] > last_processed_id:
                frame = frame_source["latest_frame"].copy()
                current_id = frame_source["frame_id"]

        if frame is not None:
            start_depth = time.time()
            small_frame = cv2.resize(frame, (640, 360))
            depth_map = depth_estimator.estimate_depth(small_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
            end_depth = time.time()
            depth_fps = 1 / (end_depth - start_depth) if (end_depth - start_depth) > 0 else 0

            with lock:
                output_dict["depth"] = depth_colored
                output_dict["depth_id"] = current_id
                output_dict["raw_depth_map"] =  depth_map
                output_dict["depth_fps"] = depth_fps
            last_processed_id = current_id
        time.sleep(0.001)

# --- Main ---
def main():
    # 1. Initialisation ROS 2 
    rclpy.init(args=None)

    shared_data = {}
    frame_source = {"frame_id" : 0}
    lock = Lock()
    stop_flag = {"stop": False}

    # Unique Node creation 
    camera_node = CameraNode(frame_source, lock)

''' Not Usefull, focal not needed

    # 2. Wait fot the Focal
    print("[INFO] Wait for the CameraInfo (focal)...")
    start_wait = time.time()
    while camera_node.fx is None:
        rclpy.spin_once(camera_node, timeout_sec=0.1)
        if time.time() - start_wait > 5.0: # Timeout of 5 secondes
            print("[WARN] No CameraInfo received after 5s. Security end")
            camera_node.fx = 550
            break

    print(f"[INFO] Focale used : {camera_node.fx}")
'''

    # 3. Initialisation of models (Now that we have fx)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # We transfer the camera_node.fx to the constructor
    depth_estimator = DepthEstimator(model_size='small', device=device, backend='depth-anything', focal_length_px=camera_node.fx)
    detector = ObjectDetector(model_size="small", conf_thres=0.1, iou_thres=0.45, device=device, depth_estimator=depth_estimator)

    # --- TensorBoard for later analysations ---
    log_dir = f"run/fire_detection_{datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard Logs saved to : {log_dir}")

    window_name = "Detection + Depth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Signal gestion
    def signal_handler(sig, frame):
        print("\n[INFO] Ctrl+C detected, clean stop...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)

    detect_thread = Thread(target=detection_worker, args=(detector, frame_source, shared_data, lock, stop_flag))
    depth_thread = Thread(target=depth_worker, args=(depth_estimator, frame_source, shared_data, lock, stop_flag))
    detect_thread.start()
    depth_thread.start()

    target_wait_ms = int(1000 / 30)

    try:
        while not stop_flag["stop"]:
            start_time = time.time()

            # 4. Primary loop 
            rclpy.spin_once(camera_node, timeout_sec=0.001)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            with lock:
                if "latest_frame" not in frame_source:
                    continue
                frame = frame_source["latest_frame"].copy()
                current_frame_id = frame_source["frame_id"]

            # --- Recovery and Display ---
            with lock:
                fallback_frame = cv2.resize(frame.copy(), (640, 360))
                detection_id = shared_data.get("detect_id", -1)
                detection_data = shared_data.get("latest_detections", None)
                detection_frame = fallback_frame

                if detection_id >= current_frame_id - TOLERANCE_DETECTION:
                    worker_result = shared_data.get("detect")
                    if worker_result is not None:
                        detection_frame = worker_result

                depth_id = shared_data.get("depth_id", -1)
                if depth_id >= current_frame_id - TOLERANCE_DEPTH:
                    depth_colored = shared_data.get("depth", None)
                else:
                    depth_colored = None

            # --- Logs ---
            status_detect = f"D: ID {detection_id}"
            status_depth = f"P: ID {depth_id}"

            if detection_id < current_frame_id - TOLERANCE_DETECTION:
                status_detect = "D: REJECTED (TOO OLD)"

            if depth_colored is None:
                status_depth = "P: REJECTED (TOO OLD)"

            print(f"FRAME READ ID {current_frame_id} | {status_detect} | {status_depth}")

            # ------------

            # Fusion and fusion
            display_frame = cv2.resize(detection_frame, (frame.shape[1], frame.shape[0]))

            if depth_colored is not None:
                if depth_colored.ndim == 2:
                    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)
                small_depth = cv2.resize(depth_colored, (frame.shape[1] // 4, frame.shape[0] // 4))
                h, w, _ = small_depth.shape
                display_frame[0:h, 0:w] = small_depth

            # --- Logs Tensorboard ---
            detect_latency = current_frame_id - detection_id
            depth_latency = current_frame_id - depth_id
            writer.add_scalar('Latency/Detection_Lag_Frames', detect_latency, current_frame_id)
            writer.add_scalar('Latency/Depth_Lag_Frames', depth_latency, current_frame_id)

            loop_end = time.time()
            loop_fps = 1 / (loop_end - start_time) if (loop_end - start_time) > 0 else 0
            writer.add_scalar('Performance/Display_FPS', loop_fps, current_frame_id)

            cv2.imshow(window_name, display_frame)

            elapsed_time_ms = (time.time() - start_time) * 1000
            delay_ms = int(target_wait_ms - elapsed_time_ms)
            if delay_ms < 1: delay_ms = 1
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in [27, ord('q')]:
                break

    finally:
        print("[INFO] Closing the code.")
        stop_flag["stop"] = True
        detect_thread.join()
        depth_thread.join()

        print("[INFO] Shutting down ROS2 nodes.")
        if rclpy.ok():
            camera_node.destroy_node() # Destroy the Unique Node
            rclpy.shutdown() # Shutdown everything at the end

        cv2.destroyAllWindows()
        print("[INFO] Complet clean shutdown.")

if __name__ == "__main__":
    main()
