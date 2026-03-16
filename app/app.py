import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
sys.path.append(r'C:\skin-classifier')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import tempfile
import os

from src.model   import get_model
from src.gradcam import predict_and_explain, LABEL_MAP_INV
from src.dataset import get_transforms

# ── Session state init ────────────────────────────────────
import pandas as pd

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load model (cached so it only loads once) ─────────────
@st.cache_resource
def load_model():
    model = get_model(dropout=0.4)
    model.load_state_dict(
        torch.load(
            r'C:\skin-classifier\models\best_model.pth',
            map_location='cpu'
        )
    )
    model.eval()
    return model

# ── Risk helper ───────────────────────────────────────────
RISK_LEVEL = {
    'Nevus'                        : ('🟢 Low Risk',    'green'),
    'Pigmented benign keratosis'   : ('🟡 Low-Medium',  'orange'),
    'Melanoma, NOS'                : ('🔴 High Risk',   'red'),
    'Basal cell carcinoma'         : ('🔴 High Risk',   'red'),
    'Squamous cell carcinoma, NOS' : ('🔴 High Risk',   'red'),
    'Dermatofibroma'               : ('🟢 Low Risk',    'green'),
    'Solar or actinic keratosis'   : ('🟡 Low-Medium',  'orange'),
}

# ── UI ────────────────────────────────────────────────────
st.title("🔬 Skin Lesion Classifier")
st.markdown("Upload a dermoscopic image to analyse the lesion.")

st.warning(
    "⚠️ This tool is for educational purposes only and is NOT a medical diagnosis. "
    "Always consult a qualified dermatologist."
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    orig_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Show uploaded image
    st.image(orig_image, caption="Uploaded Image", width='stretch')

    if st.button("🔍 Analyse"):

        # ── Image quality check ───────────────────────────
        def check_image_quality(image):
            issues = []

            # Check resolution
            h, w = image.shape[:2]
            if h < 150 or w < 150:
                issues.append("Image resolution is very low — predictions may be unreliable.")

            # Check brightness
            gray       = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            brightness = gray.mean()
            if brightness < 40:
                issues.append("Image appears very dark — try retaking in better lighting.")
            elif brightness > 220:
                issues.append("Image appears overexposed — try retaking without flash.")

            # Check blurriness
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50:
                issues.append("Image appears blurry — try holding the camera steady.")

            return issues

        issues = check_image_quality(orig_image)
        if issues:
            for issue in issues:
                st.warning(f"⚠️ Image Quality Warning: {issue}")
            st.info("The model will still attempt a prediction but results may be less reliable.")

        with st.spinner("Analysing..."):
            model     = load_model()
            transform = get_transforms('val')
            tensor    = transform(image=orig_image)['image'].unsqueeze(0)

            result = predict_and_explain(model, tensor, orig_image)

        # ── Results ──────────────────────────────────────
        risk_label, risk_color = RISK_LEVEL[result['label']]
        confidence = result['confidence']

        # Confidence threshold warning
        LOW_CONFIDENCE_THRESHOLD = 0.60
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence prediction ({confidence:.1%}). "
                f"The model is uncertain about this image. "
                f"This may be due to image quality, unusual presentation, "
                f"or a case outside the model's training distribution. "
                f"Please consult a dermatologist regardless of this result."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diagnosis",  result['label'])
            st.metric("Confidence", f"{confidence:.1%}")
        with col2:
            st.markdown("### Risk Level")
            st.markdown(
                f"<h2 style='color:{risk_color}'>{risk_label}</h2>",
                unsafe_allow_html=True
            )

        # ── ABCDE Checklist ───────────────────────────────
        st.markdown("---")
        st.subheader("📋 ABCDE Checklist")
        st.markdown("The ABCDE criteria are used by dermatologists to evaluate suspicious lesions:")

        abcde = {
            "A — Asymmetry":  "One half of the lesion doesn't match the other half in shape or color.",
            "B — Border":     "Edges are irregular, ragged, notched, or blurred rather than smooth and well-defined.",
            "C — Color":      "Multiple shades of brown, black, red, white, or blue within the same lesion.",
            "D — Diameter":   "Lesions larger than 6mm (about the size of a pencil eraser) are more concerning.",
            "E — Evolution":  "Any change in size, shape, color, or new symptoms like bleeding or itching over time."
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

        flags = ABCDE_RISK[result['label']]
        for i, (criterion, description) in enumerate(abcde.items()):
            if flags[i]:
                st.error(f"🔴 **{criterion}** — {description}")
            else:
                st.success(f"🟢 **{criterion}** — {description}")
        
        # ── Heatmap ───────────────────────────────────────
        st.markdown("---")
        st.subheader("Grad-CAM Explanation")
        st.markdown("Red areas show what influenced the prediction most.")

        col3, col4 = st.columns(2)
        with col3:
            st.image(orig_image,        caption="Original",          width='stretch')
        with col4:
            st.image(result['heatmap'], caption="Grad-CAM Heatmap",  width='stretch')

        # ── All class probabilities ───────────────────────
        st.markdown("---")
        st.subheader("All Class Probabilities")
        probs = result['probs']
        for idx, prob in enumerate(probs):
            label = LABEL_MAP_INV[idx]
            st.progress(float(prob), text=f"{label}: {prob:.1%}")
        
# ── Generate PDF Report ───────────────────────────
        def generate_pdf(orig_image, heatmap, result):
            buffer = io.BytesIO()
            doc    = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story  = []

            # Title
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#1a1a2e'),
                spaceAfter=10
            )
            story.append(Paragraph("Skin Lesion Analysis Report", title_style))
            story.append(Spacer(1, 0.3*cm))

            # Disclaimer box
            disclaimer_style = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.red,
                borderColor=colors.red,
                borderWidth=1,
                borderPadding=5,
                backColor=colors.HexColor('#fff0f0')
            )
            story.append(Paragraph(
                "⚠ DISCLAIMER: This report is for educational purposes only and is NOT a medical diagnosis. "
                "Please consult a qualified dermatologist for any skin concerns.",
                disclaimer_style
            ))
            story.append(Spacer(1, 0.5*cm))

            # Result summary table
            risk_label = RISK_LEVEL[result['label']][0]
            data = [
                ['Field',        'Value'],
                ['Diagnosis',    result['label']],
                ['Confidence',   f"{result['confidence']:.1%}"],
                ['Risk Level',   risk_label],
            ]
            table = Table(data, colWidths=[5*cm, 10*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 11),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING',    (0,0), (-1,-1), 8),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.5*cm))

            # Images side by side
            story.append(Paragraph("Visual Analysis", styles['Heading2']))
            story.append(Spacer(1, 0.2*cm))

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save original
                orig_path = os.path.join(tmpdir, 'original.jpg')
                orig_bgr  = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(orig_path, orig_bgr)

                # Save heatmap
                heat_path = os.path.join(tmpdir, 'heatmap.jpg')
                heat_bgr  = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                cv2.imwrite(heat_path, heat_bgr)

                img_table = Table([
                    [Paragraph("Original Image", styles['Normal']),
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

                # ABCDE section
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
                    color  = colors.red if flags[i] else colors.green
                    marker = "🔴" if flags[i] else "🟢"
                    story.append(Paragraph(
                        f"{marker} {item}",
                        ParagraphStyle('abcde', parent=styles['Normal'],
                                       textColor=color, fontSize=10,
                                       spaceAfter=4)
                    ))

                story.append(Spacer(1, 0.5*cm))

                # All probabilities
                story.append(Paragraph("All Class Probabilities", styles['Heading2']))
                story.append(Spacer(1, 0.2*cm))

                prob_data = [['Class', 'Probability']]
                for idx, prob in enumerate(result['probs']):
                    prob_data.append([LABEL_MAP_INV[idx], f"{prob:.1%}"])

                prob_table = Table(prob_data, colWidths=[10*cm, 5*cm])
                prob_table.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a2e')),
                    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
                    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE',      (0,0), (-1,-1), 10),
                    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
                    ('GRID',          (0,0), (-1,-1), 0.5, colors.grey),
                    ('PADDING',       (0,0), (-1,-1), 6),
                ]))
                story.append(prob_table)

                doc.build(story)

            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf(orig_image, result['heatmap'], result)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"skin_analysis_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf"
        )
        
        # ── Save to history ───────────────────────────────
        st.session_state.history.append({
            'Image'     : uploaded_file.name,
            'Diagnosis' : result['label'],
            'Confidence': f"{result['confidence']:.1%}",
            'Risk'      : RISK_LEVEL[result['label']][0]
        })
        
        # ── Disclaimer ────────────────────────────────────
        st.markdown("---")
        st.error(
            "🚨 This is NOT a medical diagnosis. "
            "If you have concerns about a skin lesion, "
            "please consult a dermatologist immediately."
        )
# ── Prediction History ────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📊 Session History")
    st.markdown("All images analysed in this session:")

    history_df = pd.DataFrame(st.session_state.history)
    history_df.index = history_df.index + 1
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()