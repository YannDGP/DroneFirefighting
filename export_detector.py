
import os
import torch
torch.cuda.init()

from ultralytics import YOLO

# 1. Chargez le modèle que vous utilisez dans votre code
# Remplacez 'yolov8s.pt' par le chemin exact de votre modèle entraîné s'il est différent.
model = YOLO('yolov8s.pt') 

# 2. Exportez-le au format 'engine' (TensorRT)
# L'argument 'half' permet d'utiliser la précision FP16 pour une vitesse maximale sur votre RTX 4060.
# L'argument 'device=0' utilise explicitement le premier GPU.
success = model.export(
    format='engine',
    half=True,  # Utilise la précision FP16
    device=0    # Utilise le GPU 0 (votre RTX 4060)
) 

if success:
    print(f"\n✅ Moteur TensorRT créé avec succès : {success}")
    print("Vous pouvez maintenant utiliser ce fichier dans votre ObjectDetector.")
else:
    print("\n❌ Échec de la création du moteur TensorRT.")