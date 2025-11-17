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
from sensor_msgs.msg import Image 

# --- Constantes de Tolérance ---
TOLERANCE_DETECTION = 3 
TOLERANCE_DEPTH = 7 

def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    x_start = (new_w - target_width) // 2
    y_start = (new_h - target_height) // 2
    return resized[y_start:y_start + target_height, x_start:x_start + target_width]

# --- Subscriber ROS2 pour récupérer les images ZED (CORRIGÉ) ---
class FrameSubscriber(Node):
    def __init__(self, frame_source, lock):
        super().__init__('frame_subscriber')
        self.frame_source = frame_source
        self.lock = lock
        self.subscription = self.create_subscription(
            Image,
            # TOPIC CORRIGÉ basé sur ros2 topic list
            '/zed/zed_node/rgb/color/rect/image',
            self.listener_callback,
            10)
        self.get_logger().info('ROS2 Frame Subscriber started, listening on /zed/zed_node/rgb/color/rect/image')

    def listener_callback(self, msg):
        # La ZED envoie probablement en BGRA (4 canaux), nous corrigeons le reshape
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape en 4 canaux (BGRA)
        try:
            # Assurez-vous d'utiliser la résolution correcte publiée par le zed_wrapper (640x360 par défaut)
            frame_4ch = image_data.reshape((msg.height, msg.width, 4))
            
            # Conversion de BGRA (4 canaux) en BGR (3 canaux) pour OpenCV
            frame = cv2.cvtColor(frame_4ch, cv2.COLOR_BGRA2BGR)
        except ValueError as e:
            self.get_logger().error(f"Erreur de Reshape: {e}. Vérifiez la taille du message : {image_data.size} vs {msg.height}x{msg.width}x3 ou 4.")
            return

        with self.lock:
            self.frame_source["latest_frame"] = frame.copy()
            self.frame_source["frame_id"] += 1

# --- Workers inchangés ---
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
            detection_frame, detection_data, _ = detector.detect(small_frame, track=False,depth_map = current_depth_map) # Récupération des données pour TensorBoard
            end_detect = time.time()

            detect_duration = end_detect - start_detect
            detect_fps = 1 / detect_duration if detect_duration >0 else 0
            # Publication du résultat (sous lock)
            with lock:
                output_dict["detect"] = detection_frame
                output_dict["detect_id"] = current_id
                output_dict["detect_fps"] = detect_fps
                output_dict["latest_detections"] = detection_data # Mise à jour des données de détection
            
            # Mise à jour de l'ID traitée UNIQUEMENT après publication réussie
            last_processed_id = current_id 

        time.sleep(0.001)
        
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

# --- Main (Adaptée à ROS 2) ---
def main():
    # --- Initialisation des modèles et TensorBoard (inchangée) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_estimator = DepthEstimator(model_size='small', device=device, backend='depth-anything')
    detector = ObjectDetector(model_size="small", conf_thres=0.1, iou_thres=0.45, device=device, depth_estimator = depth_estimator )
   
    log_dir = f"run/fire_detection_{datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard Logs saved to : {log_dir}")    
    
    window_name = "Detection + Depth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    
    shared_data = {}
    frame_source = {"frame_id" : 0}
    lock = Lock()
    stop_flag = {"stop": False}

    # --- Gestionnaire de signal (Fermeture Propre) ---
    def signal_handler(sig, frame):
        print("\n[INFO] Ctrl+C détecté, arrêt propre...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)

    detect_thread = Thread(target=detection_worker, args=(detector, frame_source, shared_data, lock, stop_flag))
    depth_thread = Thread(target=depth_worker, args=(depth_estimator, frame_source, shared_data, lock, stop_flag))
    detect_thread.start()
    depth_thread.start()

    # --- Initialisation ROS 2 ---
    rclpy.init(args=None)
    node = FrameSubscriber(frame_source, lock)
    
    # Simuler les FPS pour l'affichage uniquement (la vitesse de traitement est gérée par les threads)
    target_wait_ms = int(1000 / 30) # 30 FPS cible pour l'affichage
    
    try:
        while not stop_flag["stop"]:
            start_time = time.time()
            
            # Écoute ROS 2 pour remplir frame_source (remplace cap.read())
            rclpy.spin_once(node, timeout_sec=0.01)

            # Vérification de l'existence d'une frame et de l'arrêt de la fenêtre
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            with lock:
                if "latest_frame" not in frame_source:
                    continue # Attendre qu'une frame soit reçue
                frame = frame_source["latest_frame"].copy()
                current_frame_id = frame_source["frame_id"]

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

            # --- Enregistrement TensorBoard ---
            
            # latency and performance
            detect_latency = current_frame_id - detection_id
            depth_latency = current_frame_id - depth_id


            writer.add_scalar('Latency/Detection_Lag_Frames', detect_latency, current_frame_id)
            writer.add_scalar('Latency/Depth_Lag_Frames', depth_latency, current_frame_id)
            
            # fps des workers
            detect_fps = shared_data.get("detect_fps", None)
            depth_fps = shared_data.get("depth_fps", None)

            if detect_fps:
                writer.add_scalar('Performance/Detection_FPS', detect_fps, current_frame_id)
            if depth_fps:
                 writer.add_scalar('Performance/Depth_FPS', depth_fps, current_frame_id)


            # Metrique distance minimal
            min_distance = float('inf')

            if detection_data: # Vérifier que detection_data n'est pas None
                # Vous devez ajuster cette partie pour extraire la distance si elle est dans detection_data
                # Exemple supposé basé sur une structure [xmin, ymin, xmax, ymax, distance]
                if isinstance(detection_data, list):
                    for detection in detection_data:
                        # Assumons que detection[4] est la distance.
                        if len(detection) > 4 and detection[4] is not None and detection[4] > 0:
                             min_distance = min(min_distance, detection[4])

            if min_distance != float('inf'):
                writer.add_scalar('Ranging/Min_Detected_Distance', min_distance, current_frame_id)

            loop_end = time.time()
            loop_duration = loop_end - start_time
            loop_fps = 1 / loop_duration if loop_duration >0 else 0
            writer.add_scalar('Performance/Display_FPS', loop_fps, current_frame_id)

            cv2.imshow(window_name, display_frame)
            
            # --- LIGNE AJOUTÉE : CAPTURE D'ÉCRAN TOUTES LES 60 IMAGES ---
            if current_frame_id % 60 == 0 and current_frame_id > 0:
                screenshot_filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_frame_{current_frame_id}.png"
                cv2.imwrite(screenshot_filename, display_frame)
                print(f"[INFO] Capture d'écran enregistrée : {screenshot_filename}")
            # -------------------------------------------------------------

            elapsed_time_ms = (time.time() - start_time) * 1000
            delay_ms = int(target_wait_ms - elapsed_time_ms)
            
            if delay_ms < 1: delay_ms = 1

            key = cv2.waitKey(delay_ms) & 0xFF
        
            if key in [27, ord('q')]:
                break

    # --- Bloc de Fermeture Propre ---
    finally:
        print("[INFO] Démarrage de la procédure de fermeture propre.")
        
        # 1. Signal aux workers de s'arrêter
        stop_flag["stop"] = True
        
        # 2. Arrêt des threads de traitement
        print("[INFO] Attente de la fin des threads de worker...")
        detect_thread.join()
        depth_thread.join()
        
        # 3. Fermeture ROS 2
        print("[INFO] Fermeture du noeud ROS2 et de l'environnement rclpy.")
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
            
        # 4. Fermeture OpenCV
        cv2.destroyAllWindows()
        print("[INFO] Fermeture complète effectuée.")

if __name__ == "__main__":
    main()