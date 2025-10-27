import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
from zoedepth.utils.misc import colorize as zoe_colorize

# Add ZoeDepth to path if needed
import sys
sys.path.insert(0, "ZoeDepth")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# --- CONSTANTE DE CALIBRATION MÉTÉRIQUE ---
# Calibré pour 3.0m +- 20cm.
REAL_REF_DISTANCE = 1.0 # Mètres - Point d'ancrage métrique.

class DepthEstimator:
    def __init__(self, model_size='small', device=None, backend='depth-anything'):
        """
        Initialize the depth estimator
        """
        self.backend = backend.lower()

        # --- SUPPRESSION DE LA STABILISATION EMA ---
        self.scale_factor_ema = None 
        self.ema_alpha = 0.0 # Rendu inutile, mais laissé à 0.0 pour clarté
        # ---------------------------------------
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            # ... (initialisation du modèle inchangée) ...
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        if self.backend == 'zoedepth':
            # Load ZoeDepth
            self.conf = get_config("zoedepth_nk", model_size)
            self.model = build_model(self.conf).to(self.device).eval()
            print("Loaded ZoeDepth model")
        elif self.backend == 'depth-anything':
            # Load Depth Anything
            model_map = {
                'small': 'depth-anything/Depth-Anything-V2-Small-hf',
                'base': 'depth-anything/Depth-Anything-V2-Base-hf',
                'large': 'depth-anything/Depth-Anything-V2-Large-hf'
            }
            model_name = model_map.get(model_size.lower(), model_map['small'])
            try:
                self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device,torch_dtype=torch.float16 if "cuda" in self.device else torch.float32)
                print(f"Loaded Depth Anything v2 {model_size} model with {'FP16' if 'cuda' in self.device else 'FP32'}")
            except Exception as e:
                print(f"Failed loading Depth Anything on {self.device}, fallback to CPU: {e}")
                self.pipe = pipeline(task="depth-estimation", model=model_name, device='cpu',torch_dtype=torch.float32)
        else:
            raise ValueError("Unsupported backend. Use 'zoedepth' or 'depth-anything'")
        
    def laser_scale(self, x, y):
        """
        Retourne la distance de référence pour la mise à l'échelle.
        """
        return REAL_REF_DISTANCE


    def estimate_depth(self, image):
        """
        Estimate depth from an image (Scaled to approximate meters if using Depth-Anything)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        if self.backend == 'zoedepth':
            with torch.no_grad():
                depth = self.model.infer_pil(pil_image)
            return depth

        elif self.backend == 'depth-anything':
            result = self.pipe(pil_image)
            depth = result['depth']
            if isinstance(depth, Image.Image):
                depth = np.array(depth, dtype=np.float32)
            elif isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy().astype(np.float32)

            # --- ÉTAPE DE MISE À L'ÉCHELLE HEURISTIQUE (SANS EMA) ---
            
            # 1. Normalisation de la carte relative
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max > depth_min:
                depth_relative = (depth - depth_min) / (depth_max - depth_min)
                
                # CORRECTION : INVERSION de la carte relative pour que PROCHE = PETITE VALEUR
                depth_relative = 1.0 - depth_relative
                
                # 2. Détermination du point d'ancrage (5ème percentile pour la robustesse)
                flat_depth = depth_relative.flatten()
                depth_ref = np.percentile(flat_depth, 5) 

                # 3. Calcul du facteur de mise à l'échelle (non lissé)
                if depth_ref > 0:
                    scale_factor = REAL_REF_DISTANCE / depth_ref
                    
                    # 4. PAS DE STABILISATION. Le facteur de mise à l'échelle est la valeur brute immédiate.
                    
                    # 5. Appliquer la mise à l'échelle métrique avec le facteur immédiat
                    depth_metric = depth_relative * scale_factor
                    return depth_metric
            
            # --- ÉTAPE DE MISE À L'ÉCHELLE HEURISTIQUE (Fin) ---
            
            # Fallback
            return depth
        
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization.
        """
        
        # Normalize a copy for visualization
        depth_vis = depth_map.copy()

        # Clamper la plage (par exemple, 0 à 10 mètres) si les valeurs sont métriques
        if np.max(depth_vis) > 1.1: 
            MAX_DISTANCE_VIS = 10.0 # Mètres (limite visuelle)
            depth_vis = np.clip(depth_vis, 0, MAX_DISTANCE_VIS)
        
        depth_min = np.min(depth_vis)
        depth_max = np.max(depth_vis)
        
        if depth_max > depth_min:
            # Normalisation à [0, 1]. Ici, 0 = proche, 1 = loin.
            depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
            
            # CORRECTION COULEUR : INVERSION (1 = Proche, 0 = Loin)
            depth_vis = 1.0 - depth_vis 
            
        depth_map_uint8 = (depth_vis * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)


    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        elif method == 'p5': # 5ème percentile pour ignorer le bruit des objets proches
            return float(np.percentile(region.flatten(), 5))
        else:
            return float(np.median(region))