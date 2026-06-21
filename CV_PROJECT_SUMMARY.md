# Image Processing & Computer Vision Training Suite (OpenCV / Python)

**Role:** Author / Developer
**Tech Stack:** Python, OpenCV, NumPy, Matplotlib, scikit-learn, Ultralytics YOLOv8/YOLOv10, LaTeX

## Summary

Designed and built a comprehensive, from-scratch computer vision curriculum and codebase covering the full image-processing pipeline — from fundamental signal-processing concepts to deep-learning-based segmentation — implemented as 13 self-contained, progressively advanced Python modules with accompanying mathematical documentation (LaTeX-typeset formula reference) and an interactive HTML walkthrough.

## Key Contributions

- Implemented core digital image processing techniques in OpenCV/NumPy: sampling & quantization, color space conversion (RGB, HSV, LAB, Grayscale), histogram equalization and CLAHE-based contrast enhancement, Gaussian/median noise restoration, and morphological operations (erosion, dilation, opening, closing).
- Built classical computer vision algorithms for image segmentation and analysis: multi-method thresholding (Otsu, adaptive), edge detection (Sobel, Prewitt, Laplacian, Canny), K-Means color clustering for segmentation, and connected-component (blob) labeling/analysis.
- Implemented geometric image transformations including rotation, scaling, affine, and perspective warps.
- Integrated deep-learning-based instance segmentation and object detection using Ultralytics YOLOv8 and YOLOv10 models (nano/small/medium variants).
- Developed a real-time computer vision application ("Phone Radar"): a YOLOv8-segmentation-driven webcam pipeline that detects and masks mobile phones in the video feed, tracks multiple targets simultaneously, and renders their positions on a custom-drawn animated radar panel (rotating sweep, pulse animation, grid overlay) with configurable HSV color filtering.
- Authored a mathematical reference document (LaTeX, compiled to PDF) formalizing the algorithms used — including the Nyquist sampling criterion, quantization levels, Gaussian and Sobel kernels, and the Laplacian operator — alongside an interactive HTML guide for self-paced learning.

## Highlights

- 13 modular, independently runnable Python scripts, each isolating a single CV concept for educational clarity.
- End-to-end pipeline: classical pixel-level processing → segmentation → deep-learning object detection, in one cohesive repository.
- Real-time application combining detection, tracking, and custom visualization (radar UI built directly with OpenCV drawing primitives).
- Technical documentation written in LaTeX with rendered formulas for filters and transformations used in the codebase.
