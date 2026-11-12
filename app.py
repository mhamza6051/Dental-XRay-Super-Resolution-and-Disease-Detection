# app.py

import streamlit as st
from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ================= CONFIG =================
BASELINE_YOLO_PATH = r"models/baseline_yolo_best.pt"
SRCNN_PATH = r"models/best_srcnn.pth"

CLASS_COLORS = {
    "Cavity": "#ef4444",
    "Fillings": "#06b6d4",
    "Impacted_Tooth": "#8b5cf6",
    "Implant": "#10b981",
}

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Dental X-Ray AI Studio",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= ADVANCED UI STYLES =================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #edf5ff 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .hero-box {
        background: linear-gradient(120deg, #071b44 0%, #0b5ed7 52%, #63a7f2 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 28px;
        box-shadow: 0 20px 50px rgba(7, 27, 68, 0.22);
        margin-bottom: 1rem;
    }
    .hero-title {
        color: white;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.9rem;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.05rem;
        line-height: 1.65;
        margin-bottom: 1rem;
    }
    .chip-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .chip-green, .chip-blue, .chip-gold {
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
    }
    .chip-green {
        background: #dcfce7;
        color: #166534;
    }
    .chip-blue {
        background: #dbeafe;
        color: #1d4ed8;
    }
    .chip-gold {
        background: #fef3c7;
        color: #92400e;
    }
    .glass-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 1.2rem;
    }
    .metric-card {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.7rem 1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }
    .section-heading {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .soft-text {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .info-banner {
        background: linear-gradient(90deg, rgba(6,182,212,0.10), rgba(37,99,235,0.08));
        border-left: 4px solid #0ea5e9;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #0f172a;
    }
    .footer-note {
        text-align:center;
        color:#64748b;
        font-size:0.9rem;
        padding-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= SRCNN MODEL =================
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.features(x)


# ================= MODEL LOADERS =================
@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(path)


@st.cache_resource
def load_srcnn_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


# ================= HELPERS =================
def preprocess_for_srcnn(img: Image.Image):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor):
    arr = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_srcnn_enhancement(img: Image.Image, model: nn.Module, device: torch.device):
    tensor = preprocess_for_srcnn(img).to(device)
    with torch.no_grad():
        out = model(tensor)
    return tensor_to_pil(out)


def extract_detection_table(result):
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["Class", "Confidence", "Box"])

    names = result.names
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        conf = float(result.boxes.conf[i].item())
        xyxy = result.boxes.xyxy[i].tolist()
        rows.append(
            {
                "Class": names.get(cls_id, str(cls_id)),
                "Confidence": round(conf, 3),
                "Box": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]",
            }
        )
    return pd.DataFrame(rows)


def detection_summary_chart(df: pd.DataFrame):
    if df.empty:
        return None
    counts = df["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    fig = px.bar(
        counts,
        x="Class",
        y="Count",
        color="Class",
        text="Count",
        color_discrete_map=CLASS_COLORS,
        title="Detected Findings Distribution",
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    return fig


def build_performance_chart():
    df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "mAP@50", "mAP@50-95"],
            "Value": [0.658, 0.776, 0.809, 0.559],
        }
    )
    fig = px.bar(df, x="Metric", y="Value", text="Value", title="System Performance")
    fig.update_traces(marker_color="#156cc4")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_range=[0, 1])
    return fig


def build_conf_iou_gauge(confidence: float, iou: float):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 0.48], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#0ea5e9"}}
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=iou,
        domain={'x': [0.52, 1], 'y': [0, 1]},
        title={'text': "IoU"},
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#8b5cf6"}}
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
    return fig


# ================= SIDEBAR =================
st.sidebar.markdown("## Control Panel")
page = st.sidebar.radio(
    "Navigation",
    ["AI Dashboard", "Detection Studio", "Enhancement Preview", "Settings Guide"],
)

conf_threshold = st.sidebar.slider("YOLO confidence threshold", 0.10, 0.95, 0.25, 0.01)
iou_threshold = st.sidebar.slider("YOLO IoU threshold", 0.10, 0.95, 0.45, 0.01)
enable_enhancement = st.sidebar.toggle("Enable SRCNN enhancement preview", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Confidence threshold controls how certain the model must be before showing a prediction. "
    "IoU threshold controls how much overlap is allowed between predicted boxes."
)


# ================= LOAD MODELS =================
load_error = False
try:
    yolo_model = load_yolo_model(BASELINE_YOLO_PATH)
except Exception as e:
    st.error(f"Baseline YOLO load failed: {e}")
    load_error = True
    yolo_model = None

srcnn_model = None
srcnn_device = None
if enable_enhancement:
    try:
        srcnn_model, srcnn_device = load_srcnn_model(SRCNN_PATH)
    except Exception as e:
        st.warning(f"SRCNN preview unavailable: {e}")


# ================= HERO =================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Dental X-Ray AI Studio</div>
        <div class="hero-subtitle">DEEP SUPER-RESOLUTION AND MACHINE LEARNING-DRIVEN ENHANCEMENT OF DENTAL X-RAY IMAGES FOR EARLY DISEASE DETECTION</div>
        <div class="chip-row">
            <span class="chip-green">AI-assisted dental screening</span>
            <span class="chip-blue">Fast object detection workflow</span>
            <span class="chip-gold">Built for research and demonstration</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================= PAGE: AI DASHBOARD =================
if page == "AI Dashboard":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Precision", "0.658")
    m2.metric("Baseline Recall", "0.776")
    m3.metric("Baseline mAP@50", "0.809")
    m4.metric("SRCNN SSIM", "0.8582")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">System recommendation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="soft-text">'
            'This intelligent dental imaging system is designed to support early screening, improve workflow efficiency, '
            'and assist healthcare professionals with fast, AI-powered visual interpretation of dental radiographs. '
            'It combines automated detection with optional image enhancement to create a modern and research-ready interface.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-banner">
            Strong oral health starts with early visibility. Smart imaging tools can help reveal findings faster,
            support better clinical attention, and create a more confident diagnostic experience.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(build_performance_chart(), use_container_width=True)

    st.markdown("### Clinical Value")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="section-heading">Early Support</div><div class="soft-text">Helps surface possible dental findings quickly for earlier professional review.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="section-heading">Workflow Speed</div><div class="soft-text">Reduces manual screening burden and supports consistent image inspection.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="section-heading">Research Ready</div><div class="soft-text">Useful for AI demonstrations, academic projects, and clinical innovation showcases.</div></div>', unsafe_allow_html=True)


# ================= PAGE: DETECTION STUDIO =================
elif page == "Detection Studio":
    st.markdown("### Upload Dental X-Ray")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.markdown(
            """
            <div class="info-banner">
            Upload a dental radiograph to run automated object detection. The detector can identify findings such as cavities,
            fillings, implants, and impacted teeth.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.plotly_chart(build_conf_iou_gauge(conf_threshold, iou_threshold), use_container_width=True)

    if uploaded_file is not None and yolo_model is not None and not load_error:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        enhanced_image = None
        if enable_enhancement and srcnn_model is not None and srcnn_device is not None:
            with st.spinner("Generating enhancement preview..."):
                enhanced_image = run_srcnn_enhancement(image, srcnn_model, srcnn_device)

        with st.spinner("Running dental detection..."):
            result = yolo_model.predict(source=image_np, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            annotated = result.plot()
            annotated = annotated[:, :, ::-1]
            det_df = extract_detection_table(result)

        if enhanced_image is not None:
            c1, c2, c3 = st.columns(3)
            c1.image(image, caption="Original X-ray", use_container_width=True)
            c2.image(enhanced_image, caption="Enhanced preview", use_container_width=True)
            c3.image(annotated, caption="AI detection output", use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.image(image, caption="Original X-ray", use_container_width=True)
            c2.image(annotated, caption="AI detection output", use_container_width=True)

        st.markdown("### Detection Summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("Total detections", str(len(det_df)))
        s2.metric("Detected classes", str(det_df['Class'].nunique() if not det_df.empty else 0))
        s3.metric("Max confidence", f"{det_df['Confidence'].max():.3f}" if not det_df.empty else "0.000")

        if det_df.empty:
            st.warning("No findings were detected with the current threshold settings. Try lowering the confidence threshold.")
        else:
            col_a, col_b = st.columns([1.05, 1])
            with col_a:
                st.dataframe(det_df, use_container_width=True, hide_index=True)
            with col_b:
                fig = detection_summary_chart(det_df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)


# ================= PAGE: ENHANCEMENT PREVIEW =================
elif page == "Enhancement Preview":
    st.markdown("### Enhancement Preview Lab")
    st.markdown(
        """
        <div class="info-banner">
        This section gives a side-by-side view of the original radiograph and the enhancement preview.
        It is useful for visual inspection, academic demonstrations, and explaining how image refinement can support perception.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload image for enhancement preview", type=["jpg", "jpeg", "png"], key="enhance")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        if srcnn_model is not None and srcnn_device is not None:
            with st.spinner("Enhancing image..."):
                enhanced = run_srcnn_enhancement(image, srcnn_model, srcnn_device)
            c1, c2 = st.columns(2)
            c1.image(image, caption="Original image", use_container_width=True)
            c2.image(enhanced, caption="Enhanced image", use_container_width=True)
        else:
            st.warning("Enhancement preview is unavailable because the SRCNN model could not be loaded.")

    st.markdown("### Why image quality matters")
    st.write(
        "Clearer radiographic presentation can support better human interpretation, better teaching demonstrations, "
        "and a more intuitive understanding of dental anatomy and suspicious regions."
    )


# ================= PAGE: SETTINGS GUIDE =================
else:
    st.markdown("### YOLO Settings Guide")
    st.markdown('<div class="section-heading">Understanding the sliders</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">YOLO Confidence Threshold</div>
            <div class="soft-text">
                This controls how confident the model must be before it shows a detection.
                <ul>
                    <li><b>Lower confidence</b> shows more detections but may include false positives.</li>
                    <li><b>Higher confidence</b> shows fewer detections but with stronger certainty.</li>
                </ul>
                Recommended working range: <b>0.25 to 0.50</b>
            </div>
        </div>
        <br>
        <div class="glass-card">
            <div class="section-heading">YOLO IoU Threshold</div>
            <div class="soft-text">
                IoU controls how much overlap is allowed between bounding boxes during suppression.
                <ul>
                    <li><b>Lower IoU</b> can keep more overlapping boxes.</li>
                    <li><b>Higher IoU</b> is stricter and removes duplicate-style boxes more aggressively.</li>
                </ul>
                Recommended working range: <b>0.45 to 0.70</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Practical tips")
    tips = pd.DataFrame(
        {
            "Setting": ["Confidence 0.25", "Confidence 0.50", "IoU 0.45", "IoU 0.70"],
            "Use Case": [
                "More exploratory screening",
                "More selective detection view",
                "Balanced duplicate suppression",
                "Stricter overlap filtering",
            ],
        }
    )
    st.dataframe(tips, use_container_width=True, hide_index=True)


st.markdown('<div class="footer-note">Built with Streamlit, YOLO, and SRCNN for dental AI demonstration.</div>', unsafe_allow_html=True)
