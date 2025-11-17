import numpy as np
import cv2

class HybridDepthEstimator:
    def __init__(self, zoe_estimator, anything_estimator, threshold_m=3.0):
        self.zoe = zoe_estimator
        self.anything = anything_estimator
        self.threshold = threshold_m

    def estimate_depth(self, image):
        depth_zoe = self.zoe.estimate_depth(image)
        depth_any = self.anything.estimate_depth(image)
        mask = depth_zoe <= self.threshold
        depth_combined = np.where(mask, depth_zoe, depth_any)

        print(f"[HybridDepth] Fusion Zoe ({self.threshold}m): {np.sum(mask)} px | DepthAnything: {np.sum(~mask)} px")

        return depth_combined

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        return self.zoe.colorize_depth(depth_map, cmap)

    def get_depth_at_point(self, depth_map, x, y):
        return self.zoe.get_depth_at_point(depth_map, x, y)

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        return self.zoe.get_depth_in_region(depth_map, bbox, method)

