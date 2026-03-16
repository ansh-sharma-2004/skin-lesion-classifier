import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import io
import tempfile
import os
import pandas as pd

sys.path.append(r'C:\skin-classifier')

from src.model   import get_model
from src.gradcam import predict_and_explain, LABEL_MAP_INV
from src.dataset import get_transforms
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="DermAI — Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────
if 'history'    not in st.session_state: st.session_state.history    = []
if 'dark_mode'  not in st.session_state: st.session_state.dark_mode  = False

# ── Theme ─────────────────────────────────────────────────
dark  = st.session_state.dark_mode
bg    = "#0f1117"      if dark else "#f8fafc"
card  = "#1a1d27"      if dark else "#ffffff"
text  = "#e8eaf0"      if dark else "#0a1628"
muted = "#8b92a5"      if dark else "#4E5157"
border= "#2a2d3a"      if dark else "#e2e8f0"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {bg} !important;
        color: {text} !important;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: {bg} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: {bg} !important;
    }}

    .main .block-container {{
        padding: 0.5rem 3rem 2rem 3rem !important;
        max-width: 1400px;
    }}

    div[data-testid="stMainBlockContainer"] {{
        padding-top: 0.5rem !important;
    }}

    section[data-testid="stSidebar"] {{
        display: none !important;
    }}

    .hero-title, .hero-subtitle, p, span, div {{
        color: {text} !important;
    }}

    .diagnosis-text {{
        color: {text} !important;
    }}

    .hero-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: {text};
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }}

    .hero-subtitle {{
        font-size: 1.05rem;
        color: {muted};
        margin-bottom: 0;
    }}

    .card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}

    .section-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {muted};
        margin-bottom: 0.4rem;
    }}

    .diagnosis-text {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        font-weight: 400;
        line-height: 1.2;
        margin: 0;
    }}

    .confidence-text {{
        font-size: 1.1rem;
        color: {muted};
        margin-top: 0.2rem;
    }}

    .risk-badge-high {{
        display: inline-block;
        background: #fff1f0;
        color: #d32f2f;
        border: 1.5px solid #ffcdd2;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
    }}

    .risk-badge-medium {{
        display: inline-block;
        background: #fffbeb;
        color: #b45309;
        border: 1.5px solid #fde68a;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
    }}

    .risk-badge-low {{
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        border: 1.5px solid #bbf7d0;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
    }}

    .result-divider {{
        border: none;
        border-top: 1px solid {border};
        margin: 1rem 0;
    }}

    .abcde-flag {{
        background: {'#2d1515' if dark else '#fff1f0'};
        border-left: 3px solid #d32f2f;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        color: {'#ffb3b3' if dark else '#7f1d1d'};
        font-size: 0.92rem;
    }}

    .abcde-clear {{
        background: {'#0f2d1a' if dark else '#f0fdf4'};
        border-left: 3px solid #16a34a;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        color: {'#86efac' if dark else '#14532d'};
        font-size: 0.92rem;
    }}

    .disclaimer-bar {{
        background: {'#2a1f1f' if dark else '#fff1f0'};
        border: 1px solid #ffcdd2;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: {'#ffb3b3' if dark else '#b91c1c'};
        margin-bottom: 1.5rem;
    }}

    .stButton > button {{
        background: {'#1e2433' if dark else '#ffffff'};
        color: {'#ffffff' if dark else '#0a1628'};
        border: 1px solid {'#3b4559' if dark else '#e2e8f0'};
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}

    .stButton > button:hover {{
        background: {'#2a3550' if dark else '#f1f5f9'};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}

    .upload-area {{
        border: 2px dashed {border};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }}

      div[data-testid="stFileUploader"] {{
        background: {'#ffffff' if dark else '#ffffff'};
        border: 1.5px dashed {border} !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }}

    div[data-testid="stFileUploaderDropzone"] * {{
        color: {'#e8eaf0' if dark else '#0a1628'} !important;
        fill: {'#e8eaf0' if dark else '#0a1628'} !important;
    }}

    div[data-testid="stFileUploaderDropzone"] button {{
        background: {'#2a3550' if dark else '#f1f5f9'} !important;
        color: {'#e8eaf0' if dark else '#0a1628'} !important;
        border: 1px solid {'#3b4559' if dark else '#e2e8f0'} !important;
        border-radius: 6px !important;
    }}

    .stProgress > div > div > div > div {{
        background: {'#3b82f6' if dark else '#0a1628'} !important;
    }}

    .stProgress > div > div > div {{
        background: {'#1e2433' if dark else '#e2e8f0'} !important;
    }}

    .nav-header {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {muted};
        margin: 1.2rem 0 0.4rem 0;
    }}

    .fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0);   }}
    }}
    div[data-testid="stFileUploader"] {{
        background: {card} !important;
        border: 1.5px dashed {border} !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }}

    div[data-testid="stFileUploader"] * {{
        color: {text} !important;
    }}

    .stDownloadButton > button {{
        background: {'#1e2433' if dark else '#ffffff'} !important;
        color: {'#e8eaf0' if dark else '#0a1628'} !important;
        border: 1px solid {'#3b4559' if dark else '#e2e8f0'} !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        transition: all 0.2s !important;
    }}

    .stDownloadButton > button:hover {{
        background: {'#2a3550' if dark else '#f1f5f9'} !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
        transform: translateY(-1px) !important;
    }}

    div[data-testid="stDataFrame"] {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}

    .risk-badge-low, .risk-badge-medium, .risk-badge-high {{
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}

    div[data-testid="stDataFrame"] iframe {{
        background: {card} !important;
    }}

    .stDataFrame {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
RISK_LEVEL = {
    'Nevus'                        : ('Low Risk',    'low'),
    'Pigmented benign keratosis'   : ('Low-Medium',  'medium'),
    'Melanoma, NOS'                : ('High Risk',   'high'),
    'Basal cell carcinoma'         : ('High Risk',   'high'),
    'Squamous cell carcinoma, NOS' : ('High Risk',   'high'),
    'Dermatofibroma'               : ('Low Risk',    'low'),
    'Solar or actinic keratosis'   : ('Low-Medium',  'medium'),
}

ABCDE_RISK = {
    'Nevus'                        : [False, False, False, False, False],
    'Pigmented benign keratosis'   : [False, False, True,  False, False],
    'Melanoma, NOS'                : [True,  True,  True,  True,  True ],
    'Basal cell carcinoma'         : [False, True,  True,  False, True ],
    'Squamous cell carcinoma, NOS' : [False, True,  False, False, True ],
    'Dermatofibroma'               : [False, False, False, False, False],
    'Solar or actinic keratosis'   : [False, True,  False, False, True ],
}

LOW_CONFIDENCE_THRESHOLD = 0.60

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = get_model(dropout=0.4)
    model.load_state_dict(torch.load(
        r'C:\skin-classifier\models\best_model.pth',
        map_location='cpu'
    ))
    model.eval()
    return model

# ── Top navbar ────────────────────────────────────────────
nav1, nav2, nav3 = st.columns([3, 1, 1])
with nav1:
    st.markdown(f"""
        <div style='padding: 0.5rem 0;'>
            <span style='font-family: DM Serif Display, serif; font-size: 1.6rem; color: {text}'>DermAI</span>
            <span style='font-size: 0.75rem; color: {muted}; margin-left: 0.4rem;'>— Skin Lesion Classifier</span>
        </div>
    """, unsafe_allow_html=True)
with nav2:
    st.markdown(f"""
        <div style='font-size: 0.8rem; color: {muted}; padding-top: 0.9rem;'>
        EfficientNet-B3 · HAM10000 · 65.5% acc
        </div>
    """, unsafe_allow_html=True)
with nav3:
    dark_toggle = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.session_state.history = []
        st.rerun()
    st.markdown(f"<div style='font-size:0.7rem; color:{muted};'>Resets current session</div>", unsafe_allow_html=True)

st.markdown(f"<hr style='border:none; border-top: 1px solid {border}; margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────
st.markdown(f"""
    <div class='hero-title'>Skin Lesion Classifier</div>
    <div class='hero-subtitle'>Upload a dermoscopic image for AI-assisted analysis with visual explanation</div>
    <br>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='disclaimer-bar'>
    🔒 <b>Educational use only.</b> This tool does not provide medical diagnoses.
    Always consult a qualified dermatologist for clinical evaluation.
    </div>
""", unsafe_allow_html=True)

# Upload area
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a dermoscopic image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown(f"""
        <div class='card' style='height:100%; font-size: 0.85rem; color: {muted}; line-height: 1.8;'>
            <div class='section-label'>Supported formats</div>
            JPG, JPEG, PNG<br>
            <div class='section-label' style='margin-top:0.8rem;'>Best results</div>
            Dermoscopic images<br>
            Good lighting<br>
            Lesion centered<br>
            Min. 150×150px
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    orig_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    col_img, col_btn = st.columns([1, 1])
    with col_img:
        st.image(orig_image, caption="", width=300)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse = st.button("🔍 Analyse")

    if analyse:
        # Quality check
        def check_quality(image):
            issues = []
            h, w = image.shape[:2]
            if h < 150 or w < 150:
                issues.append("Resolution too low — predictions may be unreliable.")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if gray.mean() < 40:
                issues.append("Image too dark — try better lighting.")
            elif gray.mean() > 220:
                issues.append("Image overexposed — try without flash.")
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                issues.append("Image appears blurry — hold camera steady.")
            return issues

        issues = check_quality(orig_image)
        if issues:
            for issue in issues:
                st.warning(f"⚠️ {issue}")

        with st.spinner("Analysing..."):
            model     = load_model()
            transform = get_transforms('val')
            tensor    = transform(image=orig_image)['image'].unsqueeze(0)
            result    = predict_and_explain(model, tensor, orig_image)

        confidence   = result['confidence']
        risk_label, risk_level = RISK_LEVEL[result['label']]
        badge_class  = f"risk-badge-{risk_level}"

        # Low confidence warning
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence ({confidence:.1%}) — the model is uncertain. "
                f"Please consult a dermatologist regardless of this result."
            )

        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

        # ── Result card ───────────────────────────────────
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([2, 1, 1])

        with r1:
            st.markdown(f"""
                <div class='section-label'>Diagnosis</div>
                <div class='diagnosis-text'>{result['label']}</div>
                <div class='confidence-text'>Confidence: {confidence:.1%}</div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div class='section-label'>Risk Level</div>
                <br>
                <span class='{badge_class}'>{'🔴' if risk_level=='high' else '🟡' if risk_level=='medium' else '🟢'} {risk_label}</span>
            """, unsafe_allow_html=True)

        with r3:
            # PDF download inside result card
            def generate_pdf(orig_image, heatmap, result):
                buffer = io.BytesIO()
                doc    = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story  = []

                title_style = ParagraphStyle(
                    'Title', parent=styles['Heading1'],
                    fontSize=20, textColor=colors.HexColor('#0a1628'), spaceAfter=10
                )
                story.append(Paragraph("Skin Lesion Analysis Report", title_style))
                story.append(Spacer(1, 0.3*cm))

                disclaimer_style = ParagraphStyle(
                    'Disclaimer', parent=styles['Normal'],
                    fontSize=9, textColor=colors.red,
                    borderColor=colors.red, borderWidth=1,
                    borderPadding=5, backColor=colors.HexColor('#fff0f0')
                )
                story.append(Paragraph(
                    "DISCLAIMER: This report is for educational purposes only and is NOT a medical diagnosis. "
                    "Please consult a qualified dermatologist for any skin concerns.",
                    disclaimer_style
                ))
                story.append(Spacer(1, 0.5*cm))

                risk_label = RISK_LEVEL[result['label']][0]
                data = [
                    ['Field',      'Value'],
                    ['Diagnosis',  result['label']],
                    ['Confidence', f"{result['confidence']:.1%}"],
                    ['Risk Level', risk_label],
                ]
                table = Table(data, colWidths=[5*cm, 10*cm])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a1628')),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE',   (0,0), (-1,-1), 11),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
                    ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
                    ('PADDING',    (0,0), (-1,-1), 8),
                ]))
                story.append(table)
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Visual Analysis", styles['Heading2']))
                story.append(Spacer(1, 0.2*cm))

                with tempfile.TemporaryDirectory() as tmpdir:
                    orig_path = os.path.join(tmpdir, 'original.jpg')
                    heat_path = os.path.join(tmpdir, 'heatmap.jpg')
                    cv2.imwrite(orig_path, cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(heat_path, cv2.cvtColor(heatmap,    cv2.COLOR_RGB2BGR))

                    img_table = Table([
                        [Paragraph("Original", styles['Normal']),
                         Paragraph("Grad-CAM Heatmap", styles['Normal'])],
                        [RLImage(orig_path, width=7*cm, height=7*cm),
                         RLImage(heat_path, width=7*cm, height=7*cm)]
                    ], colWidths=[8*cm, 8*cm])
                    img_table.setStyle(TableStyle([
                        ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
                        ('PADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(img_table)
                    story.append(Spacer(1, 0.5*cm))

                    story.append(Paragraph("ABCDE Criteria", styles['Heading2']))
                    story.append(Spacer(1, 0.2*cm))
                    abcde_items = [
                        "A — Asymmetry: One half doesn't match the other.",
                        "B — Border: Irregular, ragged, or blurred edges.",
                        "C — Color: Multiple shades within the lesion.",
                        "D — Diameter: Larger than 6mm is more concerning.",
                        "E — Evolution: Any change in size, shape or color."
                    ]
                    flags = ABCDE_RISK[result['label']]
                    for i, item in enumerate(abcde_items):
                        story.append(Paragraph(
                            f"{'🔴' if flags[i] else '🟢'} {item}",
                            ParagraphStyle('abcde', parent=styles['Normal'],
                                           textColor=colors.red if flags[i] else colors.green,
                                           fontSize=10, spaceAfter=4)
                        ))

                    story.append(Spacer(1, 0.5*cm))
                    story.append(Paragraph("All Class Probabilities", styles['Heading2']))
                    story.append(Spacer(1, 0.2*cm))

                    prob_data = [['Class', 'Probability']]
                    for idx, prob in enumerate(result['probs']):
                        prob_data.append([LABEL_MAP_INV[idx], f"{prob:.1%}"])
                    prob_table = Table(prob_data, colWidths=[10*cm, 5*cm])
                    prob_table.setStyle(TableStyle([
                        ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#0a1628')),
                        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
                        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE',       (0,0), (-1,-1), 10),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
                        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
                        ('PADDING',        (0,0), (-1,-1), 6),
                    ]))
                    story.append(prob_table)
                    doc.build(story)

                buffer.seek(0)
                return buffer

            st.markdown(f"<div class='section-label'>Report</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_buffer = generate_pdf(orig_image, result['heatmap'], result)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_buffer,
                file_name=f"dermai_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Heatmap ───────────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>Grad-CAM Visual Explanation</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.95rem; color:#94a3b8; margin-bottom:0.8rem;'>Red areas show regions that most influenced the prediction</div>", unsafe_allow_html=True)
        h1, h2, h3 = st.columns([1, 1, 1])
        with h1:
            st.image(orig_image,        caption="Original",         width=300)
        with h2:
            st.image(result['heatmap'], caption="Grad-CAM Heatmap", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ABCDE ─────────────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>ABCDE Dermatology Checklist</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:{muted}; margin-bottom:0.8rem;'>Standard criteria used by dermatologists to evaluate lesions</div>", unsafe_allow_html=True)

        abcde = {
            "A — Asymmetry" : "One half of the lesion doesn't match the other half.",
            "B — Border"    : "Edges are irregular, ragged, notched, or blurred.",
            "C — Color"     : "Multiple shades of brown, black, red, white, or blue.",
            "D — Diameter"  : "Lesions larger than 6mm are more concerning.",
            "E — Evolution" : "Any change in size, shape, color, or new symptoms."
        }
        flags = ABCDE_RISK[result['label']]
        for i, (criterion, description) in enumerate(abcde.items()):
            css_class = "abcde-flag" if flags[i] else "abcde-clear"
            icon      = "🔴" if flags[i] else "🟢"
            st.markdown(
                f"<div class='{css_class}'><b>{icon} {criterion}</b> — {description}</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Probabilities ─────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>All Class Probabilities</div>", unsafe_allow_html=True)
        for idx, prob in enumerate(result['probs']):
            label = LABEL_MAP_INV[idx]
            st.progress(float(prob), text=f"{label}: {prob:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            'Image'     : uploaded_file.name,
            'Diagnosis' : result['label'],
            'Confidence': f"{confidence:.1%}",
            'Risk'      : risk_label
        })
# ── Session history ───────────────────────────────────────
if st.session_state.history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Session History</div>", unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.history)
    history_df.index = history_df.index + 1
    st.dataframe(history_df, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)