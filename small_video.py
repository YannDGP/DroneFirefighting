import cv2

input_path = "input_video.mp4"      # vidéo originale
output_path = "small_input.mp4"     # vidéo prétraitée
scale_factor = 0.5                  # réduire de moitié

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir {input_path}")

# Récupérer taille originale
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
fps = cap.get(cv2.CAP_PROP_FPS)

# Créer writer pour la nouvelle vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # réduire la taille
    small_frame = cv2.resize(frame, (width, height))
    out.write(small_frame)

cap.release()
out.release()
print(f"Vidéo prétraitée enregistrée dans {output_path}")
