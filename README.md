# Görüntü İşleme - OpenCV Eğitim Projesi

> Bu proje, görüntü işleme tekniklerini adım adım öğrenmek için hazırlanmış kapsamlı bir eğitim materyalidir. Temel kavramlardan ileri seviye deep learning tabanlı segmentasyona kadar tüm konuları içerir.

## 📁 Proje Yapısı

```
goruntu_open-main/
├── 01_goruntu_yukle.py          # Görüntü yükleme ve görüntüleme
├── 02_ornekleme_sampling.py     # Örnekleme (downsampling/upsampling)
├── 03_kuantalama_quantization.py # Kuantalama (bit derinliği)
├── 04_renk_uzaylari.py          # Renk uzayları (RGB, HSV, LAB, Grayscale)
├── 05_enhancement_histogram_clahe.py # Histogram eşitleme & CLAHE
├── 06_restorasyon_gaussian_median.py  # Gürültü azaltma (Gaussian, Median)
├── 07_morfolojik_islemler.py    # Morfolojik işlemler (erosion, dilation)
├── 08_thresholding.py           # Eşikleme (Otsu, adaptive, renkli)
├── 09_edge_detection.py         # Kenar tespiti (Sobel, Canny, Laplacian)
├── 10_kmeans_segmentasyon.py    # K-Means kümeleme ile segmentasyon
├── 11_connected_components.py   # Bağlı bileşenler analizi
├── 12_deeplearning_segmentation.py # Deep learning segmentasyon (YOLO)
├── 13_geometric_transformations.py # Geometrik dönüşümler (rotate, scale, warp)
│
├── referans.py                  # Telefon Radar uygulaması (YOLOv8)
├── ultra.py                     # YOLO detection scripti
│
├── formulas.md                  # Matematiksel formüller (LaTeX)
├── formulas.tex                 # LaTeX kaynak dosyası
├── formulas.pdf                 # Derlenmiş formüller PDF'i
│
├── anlatim.html                 # Interaktif anlatım (HTML)
├── anlatim_karma.html           # Karmaşık anlatım
│
├── Görüntü_işleme.pdf           # Ders notları PDF
│
├── yolov8n.pt                   # YOLOv8 nano model
├── yolov8s.pt                   # YOLOv8 small model
├── yolov8n-seg.pt               # YOLOv8 segmentasyon modeli
├── yolov8m-seg.pt               # YOLOv8 medium segmentasyon
├── yolov10n.pt                  # YOLOv10 nano model
├── yolov10s.pt                  # YOLOv10 small model
│
├── ornek.jpg                    # Örnek görüntü
├── harita.png                   # Harita görüntüsü
└── runs/                        # YOLO çıktıları
```

---

## 🎯 Konu Bazlı Özet

| Dosya | Konu | Açıklama |
|-------|------|----------|
| `01_goruntu_yukle.py` | Görüntü Yükleme | `cv2.imread()`, `plt.imshow()` |
| `02_ornekleme_sampling.py` | Örnekleme | Piksel sayısını azaltma/artırma |
| `03_kuantalama_quantization.py` | Kuantalama | 8-bit → 2-bit, 256 → 4 seviye |
| `04_renk_uzaylari.py` | Renk Uzayları | RGB, HSV, LAB, Grayscale dönüşümleri |
| `05_enhancement_histogram_clahe.py` | Histogram | Histogram eşitleme, CLAHE |
| `06_restorasyon_gaussian_median.py` | Gürültü Azaltma | Gaussian blur, Median blur |
| `07_morfolojik_islemler.py` | Morfoloji | Erosion, Dilation, Opening, Closing |
| `08_thresholding.py` | Eşikleme | Basit, Otsu, Adaptive thresholding |
| `09_edge_detection.py` | Kenar Tespiti | Sobel, Prewitt, Laplacian, Canny |
| `10_kmeans_segmentasyon.py` | K-Means | Renk kümeleri ile segmentasyon |
| `11_connected_components.py` | Bağlı Bileşenler | Etiketleme, blob analizi |
| `12_deeplearning_segmentation.py` | Deep Learning | YOLO ile nesne tespiti ve segmentasyon |
| `13_geometric_transformations.py` | Geometrik Dönüşümler | Rotate, Scale, Affine, Perspective |

---

## 🧮 Matematiksel Formüller

Detaylı formüller için [formulas.md](formulas.md) dosyasına bakınız.

### Temel Kavramlar

| Kavram | Formül | Açıklama |
|--------|--------|----------|
| **Dijital Görüntü** | $I(x,y) \in \mathbb{R}^{M \times N \times C}$ | M×N görüntü matrisi |
| **Nyquist** | $f_s \geq 2 \cdot f_{max}$ | Örnekleme frekansı |
| **Kuantalama** | $L = 2^b$ | Bit derinliğinden seviye sayısı |
| **Piksel Değeri** | $0 \leq I(x,y) \leq 255$ | 8-bit gri tonlama |

### Filtreler

| Filtre | Formül | Uygulama |
|--------|--------|----------|
| **Gaussian** | $G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$ | Gürültü azaltma |
| **Sobel** | $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ | Yatay kenar |
| **Laplacian** | $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ | Kenar tespiti |

---

## 🚀 Kullanım

### Gerekli Kütüphaneler

```bash
pip install opencv-python numpy matplotlib scikit-learn ultralytics
```

### Örnek Çalıştırma

```bash
# Temel görüntü işleme
python 01_goruntu_yukle.py

# Kenar tespiti
python 09_edge_detection.py

# Deep learning segmentasyon
python 12_deeplearning_segmentation.py

# Telefon radar uygulaması
python referans.py
```

---

## 📊 Referans Uygulama: Telefon Radar

[referans.py](referans.py) dosyası, yukarıdaki tekniklerin birleşimini gösteren kapsamlı bir örnektir.

### Özellikler

- **YOLOv8 Segmentasyon**: Cep telefonu tespiti
- **Radar Panel**: Gerçek zamanlı konum takibi
- **HSV Filtresi**: Renk tabanlı filtreleme (eklenmiş)
- **Multi-target**: Birden fazla telefon desteği

### Ayarlar

```python
SOURCE       = 0           # 0 = webcam, video dosya yolu
CONF         = 0.35        # Güven eşiği
HSV_ENABLE   = True       # HSV filtresi aktif
HSV_LOWER    = np.array([0, 0, 0])
HSV_UPPER    = np.array([180, 255, 255])
```

---

## 📚 Kaynaklar

- **OpenCV Dokümantasyon**: https://docs.opencv.org/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Formüller**: [formulas.pdf](formulas.pdf)

---

## Lisans

Bu proje eğitim amaçlı hazırlanmıştır.