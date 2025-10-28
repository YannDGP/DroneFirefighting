import cv2
import torch
import time
import numpy as np
from threading import Thread, Lock
from detection_model import ObjectDetector
from depth_model import DepthEstimator
import signal
import sys

# --- Constantes de Tolérance ---
# Tolérance pour la détection (le modèle plus rapide)
TOLERANCE_DETECTION = 8 
# Tolérance pour la profondeur (le modèle plus lent)
TOLERANCE_DEPTH = 18 

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
            small_frame = cv2.resize(frame, (640, 360))
            detection_frame, _, _ = detector.detect(small_frame, track=False,depth_map = current_depth_map)  
            # Publication du résultat (sous lock)
            with lock:
                output_dict["detect"] = detection_frame
                output_dict["detect_id"] = current_id
            
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
            small_frame = cv2.resize(frame, (640, 360))
            depth_map = depth_estimator.estimate_depth(small_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
            
            # Publication du résultat (sous lock)
            with lock:
                output_dict["depth"] = depth_colored
                output_dict["depth_id"] = current_id
                output_dict["raw_depth_map"] =  depth_map
            
            # Mise à jour de l'ID traitée UNIQUEMENT après publication réussie
            last_processed_id = current_id 

        time.sleep(0.001)
        #pass

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    
    #Tentative de forcer une résolution plus faible pour réduire la charge I/O et CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    
    # Initialisation des modèles
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_estimator = DepthEstimator(model_size='small', device=device, backend='depth-anything') # Utilisez 'small' si trop lent
    detector = ObjectDetector(model_size="small", conf_thres=0.1, iou_thres=0.45, device=device, depth_estimator = depth_estimator )
   
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

            # --- Récupération des résultats avec Freshness Check ---
            with lock:
                # 1. Fallback Frame
                fallback_frame = cv2.resize(frame.copy(), (640, 360))
                
                # 2. Traitement de la DÉTECTION
                detection_id = shared_data.get("detect_id", -1)
                
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
            status_detect = f"D: ID {detection_id}"
            status_depth = f"P: ID {depth_id}"
            
            # Marquer le statut de rejet si le résultat du worker n'est pas utilisé
            if detection_id < current_frame_id - TOLERANCE_DETECTION:
                status_detect = "D: REJETÉ (TROP VIEUX/FALLBACK)"
            
            if depth_colored is None:
                # depth_colored est None si rejeté OU si le worker n'a rien produit (premières frames)
                status_depth = "P: REJETÉ (TROP VIEUX/AUCUN)"
            
            print(f"FRAME LUE ID {current_frame_id} | {status_detect} | {status_depth}")
            
            # Affichage et fusion
            display_frame = cv2.resize(detection_frame, (frame.shape[1], frame.shape[0]))
            
            if depth_colored is not None:
                if depth_colored.ndim == 2:
                    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)
                
                small_depth = cv2.resize(depth_colored, (frame.shape[1] // 4, frame.shape[0] // 4))
                h, w, _ = small_depth.shape
                display_frame[0:h, 0:w] = small_depth

            cv2.imshow(window_name, display_frame)

    finally:
        # Fermeture propre
        stop_flag["stop"] = True
        detect_thread.join()
        depth_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Fermeture complète effectuée.")

if __name__ == "__main__":
    main (camera_index=0)