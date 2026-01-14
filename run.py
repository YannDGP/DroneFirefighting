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

#if torch.cuda.is_available():
 #   torch.cuda.init()
# --- Constantes de Tolérance ---
# Tolérance pour la détection (le modèle plus rapide)
TOLERANCE_DETECTION = 5 
# Tolérance pour la profondeur (le modèle plus lent)
TOLERANCE_DEPTH = 7

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
    last_processed_id = -1
    while not stop_flag["stop"]:
        frame = None
        current_id = -1
        current_depth_map = None
        
        # Acquisition : Copier UNIQUEMENT si la frame est plus récente que la dernière traitée
        with lock:
            if "latest_frame" in frame_source and frame_source["frame_id"] > last_processed_id:
                frame = frame_source["latest_frame"].copy()
                current_id = frame_source["frame_id"]

            # Récupérer la carte de profondeur la plus fraîche (pas de vérification d'ID ici, on prend la dernière)
            current_depth_map = output_dict.get("raw_depth_map",None)
        
        if frame is not None:
            # Traitement (lourd, hors du lock)
            start_detect = time.time()
            small_frame = cv2.resize(frame, (640, 360))
            detection_frame, _, _ = detector.detect(small_frame, track=False,depth_map = current_depth_map)  
            end_detect = time.time()

            detect_durarion = end_detect - start_detect
            detect_fps = 1 / detect_durarion if detect_durarion >0 else 0
            # Publication du résultat (sous lock)
            with lock:
                output_dict["detect"] = detection_frame
                output_dict["detect_id"] = current_id
                output_dict["detect_fps"] = detect_fps
            
            # Mise à jour de l'ID traitée UNIQUEMENT après publication réussie
            last_processed_id = current_id 

        time.sleep(0.001)
        #pass
        

def depth_worker(depth_estimator, frame_source, output_dict, lock, stop_flag):
    last_processed_id = -1
    while not stop_flag["stop"]:
        frame = None
        current_id = -1
        
        # Acquisition : Copier UNIQUEMENT si la frame est plus récente que la dernière traitée
        with lock:
            if "latest_frame" in frame_source and frame_source["frame_id"] > last_processed_id:
                frame = frame_source["latest_frame"].copy()
                current_id = frame_source["frame_id"]
                
        if frame is not None:
            # Traitement (lourd, hors du lock)
            start_depth = time.time()
            small_frame = cv2.resize(frame, (640, 360))
            depth_map = depth_estimator.estimate_depth(small_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
            end_depth = time.time()

            depth_duration = end_depth - start_depth
            depth_fps = 1 / depth_duration if depth_duration > 0 else 0
            # Publication du résultat (sous lock)
            with lock:
                output_dict["depth"] = depth_colored
                output_dict["depth_id"] = current_id
                output_dict["raw_depth_map"] =  depth_map
                output_dict["depth_fps"] = depth_fps
            # Mise à jour de l'ID traitée UNIQUEMENT après publication réussie
            last_processed_id = current_id 

        time.sleep(0.001)
        #pass

def main(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la source ({source}).")
        return

    
    #Tentative de forcer une résolution plus faible pour réduire la charge I/O et CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or isinstance(source, int):
        video_fps = 30
    
    target_wait_ms = int(1000 /video_fps)

    # Initialisation des modèles
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_estimator = DepthEstimator(model_size='small', device=device, backend='depth-anything') # Utilisez 'small' si trop lent
    detector = ObjectDetector(model_size="small", conf_thres=0.1, iou_thres=0.45, device=device, depth_estimator = depth_estimator )
   
    #Tensorboard (Tensor/it)
    log_dir = f"run/fire_detection_{datetime.datetime.now().strftime('%d/%m/%Y-%Hh%Mm%Ss')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard Logs saved to : {log_dir}")    
    
    # Initialisation des variables partagées et gestion des threads
    window_name = "Detection + Depth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    
    shared_data = {}
    frame_source = {"frame_id" : 0}
    lock = Lock()
    stop_flag = {"stop": False}

    def signal_handler(sig, frame):
        print("\n[INFO] Ctrl+C détecté, arrêt propre...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)

    detect_thread = Thread(target=detection_worker, args=(detector, frame_source, shared_data, lock, stop_flag))
    depth_thread = Thread(target=depth_worker, args=(depth_estimator, frame_source, shared_data, lock, stop_flag))
    detect_thread.start()
    depth_thread.start()

    try:
        while not stop_flag["stop"]:
            start_time = time.time()
            ret, frame = cap.read()
            
            
            # Gestion de la fin de la vidéo ou de la fermeture de la fenêtre
            if not ret or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

            # Stocker la frame lue
            current_frame_id = -1
            with lock:
                frame_source["latest_frame"] = frame.copy()
                frame_source["frame_id"] += 1
                current_frame_id = frame_source["frame_id"]

            detection_date = None
            # --- Récupération des résultats avec Freshness Check ---
            with lock:
                # 1. Fallback Frame
                fallback_frame = cv2.resize(frame.copy(), (640, 360))
                
                # 2. Traitement de la DÉTECTION
                detection_id = shared_data.get("detect_id", -1)
                detection_data = shared_data.get("latest_detections", None)
                
                # SÉCURITÉ (contre l'erreur cv2.resize)
                detection_frame = fallback_frame 
                
                # Vérification de fraîcheur Détection
                if detection_id >= current_frame_id - TOLERANCE_DETECTION: 
                    worker_result = shared_data.get("detect")
                    if worker_result is not None:
                        detection_frame = worker_result
                
                # 3. Traitement de la PROFONDEUR
                depth_id = shared_data.get("depth_id", -1)
                
                # Vérification de fraîcheur Profondeur
                if depth_id >= current_frame_id - TOLERANCE_DEPTH: 
                    depth_colored = shared_data.get("depth", None)
                else:
                    depth_colored = None # Masquer si trop vieux

            # --- Affichage du statut dans la console ---
            status_detect = f"DETECTION: ID {detection_id}"
            status_depth = f"DEPTH: ID {depth_id}"
            
            # Marquer le statut de rejet si le résultat du worker n'est pas utilisé
            if detection_id < current_frame_id - TOLERANCE_DETECTION:
                status_detect = "DETECTION: REFUSED (TOO OLD/FALLBACK)"
            
            if depth_colored is None:
                # depth_colored est None si rejeté OU si le worker n'a rien produit (premières frames)
                status_depth = "DEPTH: REFUSED (TOO OLD/NOTHING"
            
            print(f"FRAME READ ID {current_frame_id} | {status_detect} | {status_depth}")
            
            # Affichage et fusion
            display_frame = cv2.resize(detection_frame, (frame.shape[1], frame.shape[0]))
            
            if depth_colored is not None:
                if depth_colored.ndim == 2:
                    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)

                small_depth = cv2.resize(depth_colored, (frame.shape[1] // 4, frame.shape[0] // 4))
                h, w, _ = small_depth.shape
                display_frame[0:h, 0:w] = small_depth

            #Enregistrement TensorBoard
            
            #latency and performance
            detect_latency = current_frame_id - detection_id
            depth_latency = current_frame_id - depth_id


            writer.add_scalar('Latency/Detection_Lag_Frames', detect_latency, current_frame_id)
            writer.add_scalar('Latency/Depth_Lag_Frames', depth_latency, current_frame_id)
            
            #fps des workers
            detect_fps = shared_data.get("detect_fps", None)
            depth_fps = shared_data.get("depth_fps", None)

            if detect_fps:
                writer.add_scalar('Performance/Detection_FPS', detect_fps, current_frame_id)
            if depth_fps:
                 writer.add_scalar('Performance/Depth_FPS', depth_fps, current_frame_id)


            #Metrique distance minimal
            min_distance = float('inf')

            if detection_data and detection_data [1]:
                for detection in detection_data [1]:
                    if len(detection) > 4 and detection[4] and detection [4] is not None and detection [4] > 0:
                        min_distance = min(min_distance, detection[4])

            if min_distance != float('inf'):
                writer.add_scalar('Ranging/Min_Detected_Distance', min_distance, current_frame_id)

            loop_end = time.time()
            loop_duration = loop_end - start_time
            loop_fps = 1 / loop_duration if loop_duration >0 else 0
            writer.add_scalar('Performance/Display_FPS', loop_fps, current_frame_id)

            cv2.imshow(window_name, display_frame)

            elapsed_time_ms = (time.time() - start_time) * 1000
            delay_ms = int(target_wait_ms - elapsed_time_ms)
            
            if delay_ms < 1: delay_ms = 1

            key = cv2.waitKey(delay_ms) & 0xFF
        
            if key in [27, ord('q')]:
                break

    finally:
        # Fermeture propre
        stop_flag["stop"] = True
        detect_thread.join()
        depth_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Fermeture complète effectuée.")

if __name__ == "__main__":
    main (source='/home/jetson/Documents/Stage_Saxion/test_video/input.mp4')
	#main(source=0)
