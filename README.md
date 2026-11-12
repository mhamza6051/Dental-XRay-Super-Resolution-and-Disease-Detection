# Dental X-Ray Super-Resolution & Disease Detection

**Deep super-resolution and machine-learning-driven enhancement of dental X-ray images for early disease detection**

An end-to-end deep learning system that detects dental conditions on panoramic X-ray images with **YOLOv8**, and investigates whether **SRCNN super-resolution enhancement** can improve that detection. It ships with a trained detector, a trained super-resolution model, a full experiment notebook, and an interactive **Streamlit app — Dental X-Ray AI Studio**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8n-00d4aa)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![Task](https://img.shields.io/badge/Task-Object%20Detection%20%2B%20Super--Resolution-purple)

---

## What this project does

Dental X-rays are often low-contrast and low-resolution, which makes small findings — an early cavity, a fine filling edge — easy to miss. This project asks a practical research question:

> **Can enhancing a dental X-ray with a super-resolution network make an object detector find disease earlier and more accurately?**

To answer it, the project builds three connected stages:

1. **Detection baseline** — a YOLOv8n detector trained on 1,269 annotated dental X-rays to find 4 classes: **Cavity, Fillings, Impacted Tooth, Implant**.
2. **Super-resolution** — an SRCNN model trained on synthetically degraded (×4 downscale + blur + noise + JPEG compression) low-resolution / high-resolution image pairs.
3. **Controlled experiment** — every X-ray is enhanced with SRCNN, a second YOLOv8n detector is trained on the enhanced images under identical settings, and the two detectors are compared head-to-head.

The finished system is deployed as a **Streamlit web app** where a user uploads an X-ray and gets instant detections, an enhancement preview, and per-detection confidences.

## Key results

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **YOLOv8n — baseline (original images)** | **0.658** | **0.776** | **0.809** | **0.559** |
| YOLOv8n — trained on SRCNN-enhanced images | 0.571 | 0.501 | 0.560 | 0.305 |

| SRCNN super-resolution (test set) | Value |
|---|---|
| PSNR | **27.34 dB** |
| SSIM | **0.8582** |

### The honest finding

SRCNN measurably improved *image quality* (PSNR 27.34 dB, SSIM 0.8582) — but the detector trained on enhanced images performed **worse** than the baseline detector on original images. Super-resolution smoothed fine diagnostic detail that the detector relied on.

This negative result is reported deliberately: it shows the difference between an image that *looks* better to a human and an image that works better for a machine, and it is the most valuable scientific outcome of the project. The production app therefore ships the **baseline detector**, with SRCNN available as a visual enhancement preview.

## Dataset

Annotated dental panoramic X-rays (Roboflow-style export, CSV bounding boxes converted to YOLO format as part of this project).

| Split | Images | Annotations |
|---|---|---|
| Train | 1,075 | 8,027 |
| Validation | 121 | 780 |
| Test | 73 | 473 |

Training-set class distribution (note the real-world imbalance the model had to handle):

| Class | Instances |
|---|---|
| Fillings | 5,242 |
| Implant | 1,784 |
| Cavity | 576 |
| Impacted Tooth | 428 |

## Method

```
Annotated X-rays (CSV boxes)
        │
        ├─► Clean + convert to YOLO format ─► YOLOv8n (50 epochs, 640×640) ─► BASELINE DETECTOR
        │
        ├─► Degrade ×4 (downscale, blur, noise, JPEG q30) ─► LR/HR pairs
        │                                   └─► SRCNN (20 epochs, MSE, Adam) ─► PSNR 27.34 / SSIM 0.8582
        │
        └─► Enhance all images with SRCNN ─► YOLOv8n (same settings) ─► COMPARISON
```

- **Detection:** YOLOv8n (3.0M parameters), 50 epochs, image size 640, batch 8, trained on an NVIDIA Tesla T4.
- **Super-resolution:** SRCNN (9×9 → 5×5 → 5×5 convolutions, 64/32 filters), MSE loss, Adam (lr 1e-4), 20 epochs.
- **Evaluation:** Precision, Recall, mAP@50, mAP@50-95 for detection; PSNR and SSIM for enhancement.

## The app — Dental X-Ray AI Studio

`app.py` is a Streamlit application with four pages:

- **AI Dashboard** — headline metrics and system overview
- **Detection Studio** — upload an X-ray, adjust confidence / IoU thresholds, and get annotated detections with a results table and charts
- **Enhancement Preview** — side-by-side original vs. SRCNN-enhanced image
- **Settings Guide** — plain-language explanation of the detection thresholds

### Run it locally

```bash
git clone https://github.com/mhamza6051/Dental-XRay-Super-Resolution-and-Disease-Detection.git
cd Dental-XRay-Super-Resolution-and-Disease-Detection
pip install -r requirements.txt
streamlit run app.py
```

Trained weights are in `models/` (`baseline_yolo_best.pt` for detection, `best_srcnn.pth` for enhancement) and are loaded automatically by the app.

## Repository structure

```
├── app.py                              # Streamlit app — Dental X-Ray AI Studio
├── notebooks/
│   └── dental_xray_full_pipeline.ipynb # Full experiment: EDA → YOLO → SRCNN → comparison
├── models/
│   ├── baseline_yolo_best.pt           # Trained YOLOv8n detector (best weights)
│   └── best_srcnn.pth                  # Trained SRCNN enhancement model
├── docs/
│   └── RESULTS.md                      # Detailed results and experiment notes
├── data.yaml.example                   # YOLO dataset config template
├── requirements.txt
└── README.md
```

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · scikit-image · Streamlit · Plotly · pandas · NumPy

## Limitations & future work

- **Class imbalance** — Fillings outnumber Impacted Tooth ~12:1; class-weighted training or targeted augmentation is the next step.
- **Enhancement hurt detection** — a detection-aware (task-driven) enhancement loss, instead of pure pixel MSE, may close this gap.
- **Single dataset** — external validation on a second clinic's X-rays is needed before any clinical claim.
- Planned: stronger SR models (EDSR / SwinIR), per-class analysis for Cavity (the hardest, most important class), and Grad-CAM explainability in the app.

## Disclaimer

This is a **research and education project**. It is not a medical device and must not be used for clinical diagnosis. All outputs require review by a qualified dental professional.

## Author

**Muhammad Hamza** — Data Scientist / AI-ML Developer
GitHub: [@mhamza6051](https://github.com/mhamza6051)
