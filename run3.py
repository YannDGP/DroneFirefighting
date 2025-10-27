import cv2
import torch
import time
from threading import Thread
from depth_model import DepthEstimator

def compute_depth(frame, estimator, output_dict, key):
    depth_map = estimator.estimate_depth(frame)
    output_dict[key] = estimator.colorize_depth(depth_map)

def main():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(r'D:\Documents\Programmation\Stage_Saxion\test_video\input.mp4')
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = 1.0 / video_fps  # intervalle réel en secondes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_estimator = DepthEstimator(model_size='base', device=device, backend='depth-anything')

    cv2.namedWindow("Depth + Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth + Video", 800, 600)

    last_depth_colored = None
    depth_thread = None
    depth_dict = {}
    frame_count = 0

    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lancer le calcul depth sur une frame sur N si thread libre
        if frame_count % 1 == 0 and (depth_thread is None or not depth_thread.is_alive()):
            small_frame = cv2.resize(frame, (640, 360))
            depth_thread = Thread(target=compute_depth, args=(small_frame, depth_estimator, depth_dict, 'depth'))
            depth_thread.start()

        # Superposer la dernière depth map dispo
        if 'depth' in depth_dict:
            last_depth_colored = cv2.resize(depth_dict['depth'], (frame.shape[1]//4, frame.shape[0]//4))
        if last_depth_colored is not None:
            h, w, _ = last_depth_colored.shape
            frame[0:h, 0:w] = last_depth_colored

        cv2.imshow("Depth + Video", frame)
        frame_count += 1

        # Synchronisation avec l'horloge vidéo
        expected_time = start_time + frame_count * frame_interval
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
