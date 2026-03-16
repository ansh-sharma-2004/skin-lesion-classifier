# Skin Lesion Classifier

A deep learning application for classifying skin lesions from dermoscopic images, with Grad-CAM visual explanations to highlight regions influencing the prediction.

## Demo

Upload a dermoscopic image → get a diagnosis, confidence score, risk level, and a heatmap showing what the model focused on.

## Classes

| Label | Risk |
|---|---|
| Nevus | 🟢 Low |
| Pigmented benign keratosis | 🟡 Low-Medium |
| Melanoma | 🔴 High |
| Basal cell carcinoma | 🔴 High |
| Squamous cell carcinoma | 🔴 High |
| Dermatofibroma | 🟢 Low |
| Solar or actinic keratosis | 🟡 Low-Medium |

## Model Performance

- **Balanced Accuracy:** 65.5% across 7 classes
- **Architecture:** EfficientNet-B3 with transfer learning
- **Dataset:** HAM10000 (11,540 images after cleaning)
- **Notable:** High recall on dangerous classes — BCC 85%, Melanoma 66%, Dermatofibroma 92%

## Project Structure
```
skin-classifier/
├── data/
│   └── raw/               # ISIC HAM10000 images and metadata
├── notebooks/
│   ├── 01_eda.ipynb       # Data exploration
│   ├── 02_train.ipynb     # Training on Google Colab
│   └── 03_evaluation.ipynb# Confusion matrix, per-class metrics
├── src/
│   ├── dataset.py         # PyTorch Dataset, augmentation, dataloaders
│   ├── model.py           # EfficientNet-B3 model definition
│   ├── train.py           # Training loop with early stopping
│   └── gradcam.py         # Grad-CAM heatmap generation
├── app/
│   └── app.py             # Streamlit web application
└── models/
    └── best_model.pth     # Trained model weights
```

## Setup

**1. Clone and create environment:**
```bash
git clone <your-repo>
cd skin-classifier
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Download dataset:**
- Register at [isic-archive.com](https://isic-archive.com)
- Download HAM10000 images and metadata
- Place images in `data/raw/images/` and CSV in `data/raw/`

**3. Run the app:**
```bash
streamlit run app/app.py
```

## Training

Training was done on Google Colab with a T4 GPU. To retrain:
- Open `notebooks/02_train.ipynb` in Colab
- Follow the cells to upload data, train, and download the model

## Key Design Decisions

**Transfer learning** — EfficientNet-B3 pretrained on ImageNet. Training from scratch would require millions of images; transfer learning lets us fine-tune on 11k images effectively.

**Weighted sampling + weighted loss** — HAM10000 has severe class imbalance (67% Nevus). Both techniques ensure the model learns all 7 classes rather than just predicting the majority class.

**Grad-CAM explainability** — Black-box predictions are not acceptable in medical contexts. Grad-CAM highlights which regions of the image drove the prediction, building trust and allowing verification that the model is reasoning correctly.

**High recall priority** — In medical screening, a false negative (missing a melanoma) is far more dangerous than a false positive (unnecessary doctor visit). The model is optimized for balanced accuracy across classes rather than raw accuracy.

## Limitations

- Not a medical diagnostic tool — for educational purposes only
- Trained only on dermoscopic images, not standard phone camera photos
- Performance on rare classes (Solar keratosis, Dermatofibroma) is limited by small sample sizes
- Always consult a qualified dermatologist for any skin concerns

## Disclaimer

**This application is NOT a substitute for professional medical advice, diagnosis, or treatment.**
```


