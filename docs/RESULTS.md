# Detailed Results & Experiment Notes

All numbers below are taken directly from the experiment notebook
(`notebooks/dental_xray_full_pipeline.ipynb`) — training and evaluation were run
on Google Colab (NVIDIA Tesla T4, Ultralytics 8.4.32, PyTorch 2.10, Python 3.12).

## 1. Detection — baseline YOLOv8n (original images)

- Training: 50 epochs, imgsz 640, batch 8, pretrained `yolov8n.pt` (3,006,428 parameters, 8.1 GFLOPs)
- Final reported metrics:

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.658 | 0.776 | 0.809 | 0.559 |

An intermediate validation pass in the notebook recorded P 0.665 / R 0.740 / mAP50 0.757 / mAP50-95 0.502
(121 validation images, 780 instances). Per-class validation showed Cavity as the hardest class
(P 0.464 in that pass) — consistent with it being small, low-contrast, and under-represented.

## 2. Super-resolution — SRCNN

- LR/HR pairs were generated from the YOLO-ready images by ×4 downscaling with
  Gaussian blur, noise, and JPEG compression (quality 30) to simulate poor-quality radiographs.
- Pairs: 1,075 train / 121 validation / 73 test.
- SRCNN: Conv(3→64, 9×9) → ReLU → Conv(64→32, 5×5) → ReLU → Conv(32→3, 5×5), MSE loss, Adam lr 1e-4, 20 epochs.
- Training loss fell from 0.0334 (epoch 1) to 0.00195 (epoch 20); best validation loss 0.001896.

| Test-set PSNR | Test-set SSIM |
|---|---|
| 27.3364 dB | 0.8582 |

## 3. Detection — YOLOv8n retrained on SRCNN-enhanced images

Identical training settings to the baseline; only the input images changed
(every split enhanced with the trained SRCNN first).

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.571 | 0.501 | 0.560 | 0.305 |

## 4. Conclusion

Enhancement improved pixel-level image quality but **reduced detection accuracy on every metric**.
Likely causes: MSE-trained SRCNN smooths high-frequency detail (edges of small cavities, fine
filling margins) that the detector uses as evidence, and enhancement artefacts shift the image
distribution away from what the detector architecture expects.

Practical takeaway shipped in this repo: **the app uses the baseline detector for diagnosis-support
detections and offers SRCNN only as a human-facing visual preview.** Future work should optimise
enhancement with a detection-aware loss rather than pixel fidelity alone.
