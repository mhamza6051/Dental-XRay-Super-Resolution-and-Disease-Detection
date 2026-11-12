# Trained Model Weights

| File | Model | Purpose | Size |
|---|---|---|---|
| `baseline_yolo_best.pt` | YOLOv8n (Ultralytics) | Dental finding detection — 4 classes: Cavity, Fillings, Impacted_Tooth, Implant | ~6 MB |
| `best_srcnn.pth` | SRCNN (PyTorch) | ×4 super-resolution enhancement of dental X-rays | ~274 KB |

Both files are loaded automatically by `app.py`. They are the best checkpoints from the
training runs documented in `notebooks/dental_xray_full_pipeline.ipynb`
(see `docs/RESULTS.md` for the full metrics).
