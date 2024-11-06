import cv2
import numpy as np

# Загрузка каскада для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения маски
mask_image = cv2.imread('haloween.png', cv2.IMREAD_UNCHANGED)
mask_h, mask_w, _ = mask_image.shape


def resize_mask(mask, new_size):
    return cv2.resize(mask, new_size)


def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    if x >= bg_w or y >= bg_h:
        return background
    h, w = overlay.shape[:2]
    if x + w > bg_w:
        w = bg_w - x
        overlay = overlay[:, :w]
    if y + h > bg_h:
        h = bg_h - y
        overlay = overlay[:h]
    if overlay.shape[2] >= 4:
        src_alpha = overlay[:, :, 3] / 255.0
        alpha = src_alpha[..., np.newaxis].repeat(3, axis=-1)
        result = background[y:y + h, x:x + w] * (1 - alpha) + overlay * alpha
    else:
        result = overlay
    background[y:y + h, x:x + w] = result
    return background


def apply_mask_to_faces(frame):
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        # Ресайз маски под размер лица
        resized_mask = resize_mask(mask_image, (w, h))

        # Наложение маски на лицо
        frame[y:y + h, x:x + w] = overlay_transparent(frame[y:y + h, x:x + w], resized_mask, 0, 0)

    return frame


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


# Открытие камеры (обычно 0 - это основная камера)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = apply_mask_to_faces(frame)

    # Вывод обработанного кадра
    cv2.imshow('Live Stream with Mask', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
