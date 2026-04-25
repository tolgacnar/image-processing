"""
Telefon Radar
=============
Gerçek zamanlı cep telefonu tespiti + radar paneli.

- YOLOv8 segmentasyon modeli kullanır (yolov8n-seg.pt)
- Cep telefonu algılandığında:
    * Ana ekranda telefon bölgesi segmentasyonla maskelenir / highlight edilir
    * Yanında siyah radar paneli açılır
    * Telefonun ekrandaki konumuna göre radar'da kırmızı nokta + pulse animasyonu çizilir
    * Birden fazla telefon varsa hepsi ayrı nokta olarak gösterilir

Kurulum:
    pip install ultralytics opencv-python numpy

Çalıştır:
    python telefon_radar.py
"""

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

# ── Ayarlar ───────────────────────────────────────────────────────────────────
SOURCE       = 0           # 0 = webcam, ya da video dosya yolu
CONF         = 0.35        # Güven eşiği
PHONE_CLASS  = 67          # COCO class 67 = cell phone
CAM_W        = 640
CAM_H        = 480
PANEL_W      = 320         # Siyah radar paneli genişliği
PANEL_H      = CAM_H
# HSV Filtre Ayarları
HSV_ENABLE   = True         # HSV filtresi aktif/pasif
HSV_LOWER    = np.array([0, 0, 0])      # Alt eşik (H, S, V)
HSV_UPPER    = np.array([180, 255, 255]) # Üst eşik (H, S, V)
# ─────────────────────────────────────────────────────────────────────────────

# Renkler
RED         = (0, 0, 255)
GREEN       = (0, 255, 80)
DARK_GREEN  = (0, 180, 50)
WHITE       = (255, 255, 255)
GRAY        = (120, 120, 120)
ORANGE      = (0, 165, 255)
PANEL_BG    = (10, 10, 10)


def draw_radar_panel(panel, phone_centers, cam_w, cam_h, pulse_phase):
    """
    Siyah radar panelini çiz.
    phone_centers: [(cx, cy), ...] — kamera koordinatları (0..cam_w, 0..cam_h)
    pulse_phase  : animasyon fazı (sürekli artan float)
    """
    h, w = panel.shape[:2]
    panel[:] = (12, 12, 12)  # Koyu arka plan

    cx, cy = w // 2, h // 2

    # ── Grid çizgileri ──
    grid_color = (30, 30, 30)
    step = 40
    for x in range(0, w, step):
        cv2.line(panel, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, step):
        cv2.line(panel, (0, y), (w, y), grid_color, 1)

    # ── Radar halkası ──
    max_r = min(cx, cy) - 20
    for i in range(1, 5):
        r = int(max_r * i / 4)
        cv2.circle(panel, (cx, cy), r, (0, 60, 0), 1)

    # ── Çapraz eksenler ──
    cv2.line(panel, (cx, 0),  (cx, h),  (0, 50, 0), 1)
    cv2.line(panel, (0, cy),  (w, cy),  (0, 50, 0), 1)

    # ── Dönen tarama çizgisi ──
    scan_angle = (pulse_phase * 60) % 360  # derece/saniye
    rad = math.radians(scan_angle)
    ex  = int(cx + max_r * math.cos(rad))
    ey  = int(cy + max_r * math.sin(rad))
    cv2.line(panel, (cx, cy), (ex, ey), (0, 100, 0), 2)

    # Tarama izi (fade)
    for fade in range(1, 30):
        a = math.radians(scan_angle - fade * 2)
        ex2 = int(cx + max_r * math.cos(a))
        ey2 = int(cy + max_r * math.sin(a))
        intensity = int(80 * (1 - fade / 30))
        cv2.line(panel, (cx, cy), (ex2, ey2), (0, intensity, 0), 1)

    # ── Merkez nokta ──
    cv2.circle(panel, (cx, cy), 4, DARK_GREEN, -1)

    # ── Başlık ──
    cv2.putText(panel, "RADAR", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(panel, f"Hedef: {len(phone_centers)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

    # ── Telefon noktaları ──
    for i, (pcx, pcy) in enumerate(phone_centers):
        # Kamera koordinatını radar koordinatına normalize et
        rx = int((pcx / cam_w) * w)
        ry = int((pcy / cam_h) * h)

        # Pulse animasyonu — her hedef farklı faz offset
        pulse_r = int(DOT_BASE_R + DOT_PULSE_AMP * math.sin(pulse_phase * 4 + i * 1.5))
        pulse_r = max(4, pulse_r)

        # Dış parlama (glow)
        for g in range(4, 0, -1):
            alpha_val = 60 - g * 12
            glow_color = (0, 0, max(0, alpha_val))
            cv2.circle(panel, (rx, ry), pulse_r + g * 4, glow_color, 1)

        # Kırmızı nokta
        cv2.circle(panel, (rx, ry), pulse_r, RED, -1)
        cv2.circle(panel, (rx, ry), pulse_r + 2, (0, 0, 180), 1)

        # Koordinat etiketi
        label = f"#{i+1} ({pcx},{pcy})"
        lx = min(rx + 10, w - 100)
        ly = max(ry - 8, 15)
        cv2.putText(panel, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ORANGE, 1)

        # Artı işareti
        cv2.line(panel, (rx - 12, ry), (rx + 12, ry), RED, 1)
        cv2.line(panel, (rx, ry - 12), (rx, ry + 12), RED, 1)

    # ── Zaman damgası ──
    ts = time.strftime("%H:%M:%S")
    cv2.putText(panel, ts, (w - 75, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    return panel


DOT_BASE_R   = 10
DOT_PULSE_AMP = 5


def apply_hsv_filter(frame, hsv_lower, hsv_upper):
    """
    HSV renk uzayında filtre uygula.
    Belirtilen aralıktaki renkleri beyaz, diğerlerini siyah yapar.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    # Morfolojik işlemler ile gürültü azalt
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_phone_overlay(frame, box, mask_data, idx):
    """
    Ana kamera görüntüsünde telefonu highlight et.
    Segmentasyon maskesi varsa kullan, yoksa bbox çiz.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Segmentasyon maskesi
    if mask_data is not None:
        seg_mask = mask_data.data[0].cpu().numpy().astype(np.uint8) * 255
        seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
        colored  = np.zeros_like(frame)
        colored[:, :] = (0, 0, 200)
        alpha_mask = (seg_mask / 255.0 * 0.4)[:, :, np.newaxis]
        frame[:] = (frame * (1 - alpha_mask) + colored * alpha_mask).astype(np.uint8)
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, RED, 2)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)

    # Etiket
    label = f"TELEFON #{idx+1}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), RED, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)

    # Merkez işareti
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), RED,
                   cv2.MARKER_CROSS, 20, 2)


def main():
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("[!] Kamera açılamadı!")
        return

    # Segmentasyon modeli
    print("[*] Model yükleniyor: yolov8n-seg.pt ...")
    model = YOLO("yolov8n-seg.pt")
    print("[*] Hazır! Q ile çıkın.")

    start_time   = time.time()
    panel_visible = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── HSV Filtresi ──
        if HSV_ENABLE:
            hsv_mask = apply_hsv_filter(frame, HSV_LOWER, HSV_UPPER)
            # Maskeyi görüntülemek için alpha blending
            hsv_display = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
            # HSV maskesini sağ panelde göster (opsiyonel)
            # frame = cv2.addWeighted(frame, 0.7, hsv_display, 0.3, 0)

        pulse_phase = time.time() - start_time

        # YOLO segmentasyon tahmini
        results = model.predict(frame, conf=CONF, verbose=False, classes=[PHONE_CLASS])

        phone_centers = []
        output = frame.copy()

        boxes = results[0].boxes
        masks = results[0].masks  # Segmentasyon maskeleri (None olabilir)

        if boxes is not None and len(boxes):
            panel_visible = True
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                phone_centers.append((cx, cy))

                # Her telefon için ayrı maske (varsa)
                mask_i = None
                if masks is not None and i < len(masks.data):
                    class_masks = type('M', (), {'data': masks.data[i:i+1]})()
                    mask_i = class_masks

                draw_phone_overlay(output, box, mask_i, i)
        else:
            panel_visible = False

        # ── Bilgi çubuğu ──
        bar_h = 30
        cv2.rectangle(output, (0, 0), (CAM_W, bar_h), (20, 20, 20), -1)
        telefon_txt = f"Telefon: {len(phone_centers)}"
        color_txt   = RED if phone_centers else GRAY
        cv2.putText(output, telefon_txt, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 2)
        cv2.putText(output, "Q:Cikis", (CAM_W - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)

        # ── Radar paneli ──
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        if panel_visible:
            draw_radar_panel(panel, phone_centers, CAM_W, CAM_H, pulse_phase)
            # Panel ayracı
            cv2.line(panel, (0, 0), (0, PANEL_H), (0, 80, 0), 2)
        else:
            # Bekleme ekranı
            panel[:] = (12, 12, 12)
            cv2.putText(panel, "RADAR", (PANEL_W//2 - 35, PANEL_H//2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 0), 2)
            cv2.putText(panel, "Bekleniyor...", (PANEL_W//2 - 60, PANEL_H//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 40, 0), 1)
            # Yavaş dönen boş radar
            draw_radar_panel(panel, [], CAM_W, CAM_H, pulse_phase)

        # ── İki ekranı yan yana birleştir ──
        combined = np.hstack([output, panel])

        cv2.imshow("Telefon Radar", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Kapatıldı.")


if __name__ == "__main__":
    main()