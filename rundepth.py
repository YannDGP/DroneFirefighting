import cv2
import torch
import time
import numpy as np
from threading import Thread, Lock
from depth_model import DepthEstimator
import signal

def depth_worker(depth_estimator, frame_source, output_dict, lock, stop_flag):
    while not stop_flag["stop"]:
        frame = None
        with lock:
            if "latest_frame" in frame_source:
                frame = frame_source["latest_frame"].copy()
        if frame is not None:
            small_frame = cv2.resize(frame, (640, 360))
            depth_map = depth_estimator.estimate_depth(small_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
            with lock:
                output_dict["depth"] = depth_colored

def main(video_path=None):
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo / caméra.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = 1.0 / video_fps

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Utilisation de : {device}")

    depth_estimator = DepthEstimator(model_size='base', device=device, backend='depth-anything')

    cv2.namedWindow("Depth Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Only", 960, 540)

    shared_data = {}
    frame_source = {}
    lock = Lock()
    stop_flag = {"stop": False}

    # --- Ctrl+C ---
    def signal_handler(sig, frame):
        print("\n[INFO] Ctrl+C détecté, arrêt propre...")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, signal_handler)

    # Thread de profondeur
    depth_thread = Thread(target=depth_worker, args=(depth_estimator, frame_source, shared_data, lock, stop_flag))
    depth_thread.start()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la vidéo détectée.")
            stop_flag["stop"] = True
            break

        # Stocker frame pour le thread
        with lock:
            frame_source["latest_frame"] = frame.copy()

        # Afficher la carte de profondeur si dispo
        with lock:
            depth_colored = shared_data.get("depth", None)

        if depth_colored is not None:
            display_frame = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            cv2.imshow("Depth Only", display_frame)

        # FPS sync
        frame_count += 1
        expected_time = start_time + frame_count * frame_interval
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            stop_flag["stop"] = True
            break

    depth_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_path=r'D:\\Documents\\Programmation\\Stage_Saxion\\test_video\\input.mp4')
