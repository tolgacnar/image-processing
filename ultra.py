"""
Görünmezlik Pelerini v5
=======================
- Faz 1: Arka planı kaydet
- Faz 2: İnsan dışı her nesne algılanırsa o bölgeyi arka planla sil

Kurulum:  pip install ultralytics opencv-python numpy
Çalıştır: python gorünmezlik_v5.py
"""

import cv2
import numpy as np
from ultralytics import YOLO

SOURCE       = 0
BG_FRAMES    = 60
PERSON_CLASS = 0
CONF         = 0.4
MORPH_SIZE   = 21
BLUR_SIZE    = 31
BLEND_ALPHA  = 0.95


def capture_background(cap):
    frames = []
    print("[FAZ 1] Arka plan kaydediliyor — kamera önünü BOŞ bırakın!")
    for i in range(BG_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))

        display = frame.copy()
        pct     = int((i / BG_FRAMES) * 100)
        bar_w   = 400
        filled  = int(bar_w * i / BG_FRAMES)
        cv2.rectangle(display, (120, 200), (120 + bar_w, 240), (50, 50, 50), -1)
        cv2.rectangle(display, (120, 200), (120 + filled, 240), (0, 220, 100), -1)
        cv2.putText(display, f"Arka plan kaydediliyor... %{pct}",
                    (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "KAMERAYI BOS BIRAKIN!",
                    (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        cv2.imshow("Gorünmezlik Pelerini v5", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[FAZ 1] Hazır!")
    return np.median(frames, axis=0).astype(np.uint8)


def apply_mask(frame, background, boxes):
    if not boxes:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask_soft = cv2.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)

    alpha  = mask_soft.astype(np.float32) / 255.0 * BLEND_ALPHA
    alpha3 = alpha[:, :, np.newaxis]
    result = (background.astype(np.float32) * alpha3 +
              frame.astype(np.float32)      * (1.0 - alpha3))
    return result.astype(np.uint8)


def main():
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[!] Kamera açılamadı!")
        return

    background = capture_background(cap)
    model      = YOLO("yolov8n.pt")
    show_debug = False

    print("[FAZ 2] Aktif! İnsan dışı nesne algılananı siler.")
    print("        Q:Çıkış | D:Debug | B:BG yenile")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF, verbose=False)

        erase_boxes = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    continue  # İnsanı atla
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                erase_boxes.append((x1, y1, x2, y2))

        output = apply_mask(frame, background, erase_boxes)

        if show_debug:
            for (x1, y1, x2, y2) in erase_boxes:
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        status = f"Silinen nesne: {len(erase_boxes)}" if erase_boxes else "Bekleniyor..."
        color  = (0, 100, 255) if erase_boxes else (180, 180, 180)
        cv2.putText(output, status,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(output, "Q:Cikis  D:Debug  B:BG yenile",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Gorünmezlik Pelerini v5", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('b'):
            background = capture_background(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


