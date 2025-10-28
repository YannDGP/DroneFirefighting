import cv2
import torch
import time
from threading import Thread, Lock
from detection_model import ObjectDetector
from depth_model import DepthEstimator
import signal

def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    # Crop centré
    x_start = (new_w - target_width) // 2
    y_start = (new_h - target_height) // 2
    return resized[y_start:y_start + target_height, x_start:x_start + target_width]

def detection_worker(detector, frame_source, output_dict, lock, stop_flag):
    while not stop_flag["stop"]:
        frame = None
        with lock:
            frame = frame_source.get("latest_frame", None)
            if frame is not None:
                frame = frame.copy()
        if frame is not None:
            #small_frame = cv2.resize(frame, (600, 360))
            small_frame = resize_with_aspect_ratio(frame, 600, 360)
            detection_frame, _, _ = detector.detect(small_frame, track=False)
            with lock:
                output_dict["detect"] = detection_frame
        time.sleep(0.001)

def depth_worker(depth_estimator, frame_source, output_dict, lock, stop_flag):
    while not stop_flag["stop"]:
        frame = None
        with lock:
            frame = frame_source.get("latest_frame", None)
            if frame is not None:
                frame = frame.copy()
        if frame is not None:
            small_frame = resize_with_aspect_ratio(frame, 600, 360)
            #small_frame = cv2.resize(frame, (600, 360))
            depth_map = depth_estimator.estimate_depth(small_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
            with lock:
                output_dict["depth_raw"] = depth_map
                output_dict["depth"] = depth_colored
        time.sleep(0.001)

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = 1.0 / video_fps

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device utilisé : {device}")
    detector = ObjectDetector(model_size="nano", conf_thres=0.1, iou_thres=0.45, device=device)
    depth_estimator = DepthEstimator(model_size='base', device=device, backend='depth-anything')

    cv2.namedWindow("Detection + Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection + Depth", 600, 360)

    shared_data = {}
    frame_source = {}
    lock = Lock()
    stop_flag = {"stop": False}

    # Ctrl+C handler
    def signal_handler(sig, frame):
        print("\n[INFO] Ctrl+C détecté, arrêt...")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, signal_handler)

    # Lancer les threads
    detect_thread = Thread(target=detection_worker, args=(detector, frame_source, shared_data, lock, stop_flag))
    depth_thread = Thread(target=depth_worker, args=(depth_estimator, frame_source, shared_data, lock, stop_flag))
    detect_thread.start()
    depth_thread.start()

    try:
        while not stop_flag["stop"]:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Impossible de lire la caméra.")
                break

            if cv2.getWindowProperty("Detection + Depth", cv2.WND_PROP_VISIBLE) < 1:
                stop_flag["stop"] = True
                break

            with lock:
                frame_source["latest_frame"] = frame.copy()
                detection_frame = shared_data.get("detect", cv2.resize(frame.copy(), (640, 360)))
                depth_colored = shared_data.get("depth", None)
                depth_raw = shared_data.get("depth_raw", None)

            display_frame = cv2.resize(detection_frame, (frame.shape[1], frame.shape[0]))
            
            if depth_raw is not None:
                boxes = detector.get_last_boxes()
                if boxes :
                    detect_h, detect_w = detection_frame.shape[:2]
                    depth_h, depth_w = depth_raw.shape[:2]
                    scale_x = depth_w / detect_w
                    scale_y = depth_h / detect_h       
                    
                    for box in boxes:
                        x1,y1,x2,y2= box
                        scale_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        distance = depth_estimator.get_depth_in_region(depth_raw, scaled_box, method='median')

                        x1_disp, y1_disp = int(x1), int(y1)
                        cv2.putText(display_frame, f"{distance:.2f} m", (x1_disp, y1_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            if depth_colored is not None:
                small_depth = cv2.resize(depth_colored, (frame.shape[1] // 4, frame.shape[0] // 4))
                h, w, _ = small_depth.shape
                display_frame[0:h, 0:w] = small_depth

            cv2.imshow("Detection + Depth", display_frame)

            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                stop_flag["stop"] = True
                break

            time.sleep(frame_interval)

    finally:
        stop_flag["stop"] = True
        detect_thread.join()
        depth_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Fermeture complète effectuée.")

if __name__ == "__main__":
    main(camera_index=0)
