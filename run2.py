import cv2
from detection_model import ObjectDetector
import torch

def main():
    cap = cv2.VideoCapture(r'D:\Documents\Programmation\Stage_Saxion\test_video\input.mp4')
    detector = ObjectDetector(model_size="nano", conf_thres=0.1, iou_thres=0.45,
                              classes=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    frame_count = 0
    last_detection_frame = None  # Pour stocker la dernière frame analysée

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # On ne fait la détection que sur 1 frame sur 2
        if frame_count % 3 == 0:
            detection_frame, detections, _ = detector.detect(frame.copy(), track=False)
            last_detection_frame = detection_frame
        else:
            detection_frame = last_detection_frame  # réutilise la dernière frame détectée

        cv2.imshow("Detection", detection_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
