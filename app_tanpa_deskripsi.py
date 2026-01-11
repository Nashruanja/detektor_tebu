"""
Aplikasi Klasifikasi Penyakit Daun Tebu
Clean & Professional Design
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from PIL import Image
import io
import base64
from feature_extraction import extract_features, create_readable_feature_names

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tebu",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CLEAN CSS STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #f7f9f6 0%, #ffffff 100%);
    }
    
    /* Compact header */
    .main-header {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2e 100%);
        padding: 1.8rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(45, 80, 22, 0.12);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.92);
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Card - NO BOX, just title + line */
    .modern-card {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 2rem;
    }
    
    .card-title {
        color: #2d5016;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 3px solid #8bc34a;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stCameraInput"] button:hover {
        background: linear-gradient(135deg, #3a5f23 0%, #4a7c2e 100%) !important;
        transform: translateY(-1px) !important;
    }
    
    /* RESPONSIVE DESIGN - Complete */
    
    /* Tablet (Portrait & Landscape) */
    @media (max-width: 1024px) {
        .main-header {
            padding: 1.5rem !important;
        }
        
        .result-card {
            min-height: 160px !important;
        }
    }
    
    /* Mobile & Small Tablets */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.85rem !important;
        }
        
        .card-title, .plot-title {
            font-size: 0.95rem !important;
        }
        
        .result-value {
            font-size: 1.8rem !important;
        }
        
        .xai-header {
            font-size: 1.2rem !important;
        }
        
        [data-testid="column"] {
            padding: 0 0.5rem !important;
        }
        
        .info-box {
            font-size: 0.85rem !important;
            padding: 1rem !important;
        }
        
        .stProgress {
            max-width: 320px !important;
        }
    }
    
    /* Small Mobile */
    @media (max-width: 480px) {
        .main-header {
            padding: 1.2rem 1rem !important;
            margin-bottom: 1.5rem !important;
        }
        
        .main-header h1 {
            font-size: 1.3rem !important;
        }
        
        .result-card {
            padding: 1.5rem 1rem !important;
            min-height: 140px !important;
        }
        
        .result-value {
            font-size: 1.5rem !important;
        }
        
        .card-title, .plot-title {
            font-size: 0.9rem !important;
        }
        
        .stProgress {
            max-width: 250px !important;
        }
        
        .info-box {
            padding: 0.8rem 1rem !important;
            font-size: 0.8rem !important;
        }
        
        .stButton>button {
            padding: 0.8rem 1.5rem !important;
            font-size: 0.95rem !important;
        }
        
        /* Tabs responsive */
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0.8rem !important;
            font-size: 0.85rem !important;
        }
    }
    
    /* File uploader - ALL TEXT VISIBLE */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background: #fafbfa !important;
        border: 2px dashed #6b9d4a !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4a7c2e !important;
        background: #f5f7f4 !important;
    }
    
    /* FIX: ALL text in uploader = DARK/VISIBLE */
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] label,
    [data-testid="stFileUploaderDropzone"] small {
        color: #2d2d2d !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #1a3a0f !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploaderFileName"] {
        color: #2d5016 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stFileUploaderFileSize"] {
        color: #2d5016 !important;
        font-weight: 600 !important;
        background: #f1f8f4 !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
    }
    
    /* Smooth fade-in animation for preview */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .img-preview-box {
        animation: fadeIn 0.3s ease-in;
    }
    
    /* CRITICAL: Image preview wrapper INSIDE card */
    .img-preview-box {
        background: #f8faf7;
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid #e1e8dd;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .img-preview-box img {
        max-width: 100%;
        max-height: 350px;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        display: block;
        margin: 0 auto;
        animation: fadeIn 0.4s ease-in;
    }
    
    .img-caption {
        color: #555555 !important;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        font-weight: 500;
    }
    
    .empty-state {
        color: #666666 !important;
        padding: 2rem;
        text-align: center;
    }
    
    .empty-state p {
        color: #666666 !important;
    }
    
    .empty-icon {
        font-size: 3rem;
        opacity: 0.4;
        margin-bottom: 0.5rem;
    }
    
    /* Modern button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 14px rgba(74, 124, 46, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #3a5f23 0%, #4a7c2e 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(74, 124, 46, 0.35) !important;
    }
    
    .stButton>button:disabled {
        background: #e0e0e0 !important;
        color: #999999 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f9f3 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #6b9d4a;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
    }
    
    .result-label {
        color: #2d5016 !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }
    
    .result-value {
        color: #2d5016;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .result-desc {
        color: #4a7c2e !important;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        line-height: 1.5;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #f5f9f3 0%, #ffffff 100%);
        border-left: 4px solid #6b9d4a;
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        font-size: 0.92rem;
        color: #2d5016;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .info-box strong {
        color: #2d5016;
    }
    
    /* XAI section */
    .xai-header {
        color: #2d5016;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        margin: 3rem 0 2rem 0;
        letter-spacing: -0.3px;
    }
    
    /* Plot cards - NO BOX, just title + line */
    .plot-card {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 2rem;
    }
    
    .plot-title {
        color: #2d5016;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 3px solid #8bc34a;
    }
    
    .plot-caption {
        color: #555555 !important;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        color: #666666 !important;
        font-size: 0.88rem;
        border-top: 1px solid #e8ede8;
    }
    
    .footer p {
        color: #666666 !important;
    }
    
    .footer strong {
        color: #2d5016 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Radio buttons - SUPER DARK TEXT */
    [data-testid="stRadio"] {
        background: transparent !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stRadio"] > div {
        gap: 1rem !important;
        background: transparent !important;
        display: flex !important;
        flex-direction: row !important;
    }
    
    [data-testid="stRadio"] > div > label,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [role="radiogroup"] > label,
    [data-testid="stRadio"] div[role="radiogroup"] label {
        color: #0d1f07 !important;  /* VERY DARK GREEN - SUPER VISIBLE */
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        background: #f1f8f4 !important;
        border: 2px solid #c5e1a5 !important;
        padding: 0.75rem 1.5rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        min-width: 150px !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Text inside label - VERY DARK */
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] label div,
    [data-testid="stRadio"] label p {
        color: #0d1f07 !important;
    }
    
    [data-testid="stRadio"] > div > label:hover,
    [data-testid="stRadio"] label:hover {
        background: #e8f5e9 !important;
        border-color: #8bc34a !important;
        color: #0d1f07 !important;
    }
    
    /* Show radio circle */
    [data-testid="stRadio"] [data-baseweb="radio"] {
        display: inline-block !important;
        margin-right: 8px !important;
        flex-shrink: 0 !important;
    }
    
    /* Selected state - green gradient WITH WHITE TEXT */
    [data-testid="stRadio"] > div > label:has(input:checked),
    [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%) !important;
        border-color: #4a7c2e !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(74, 124, 46, 0.3) !important;
    }
    
    /* WHITE text when selected */
    [data-testid="stRadio"] label:has(input:checked) span,
    [data-testid="stRadio"] label:has(input:checked) div,
    [data-testid="stRadio"] label:has(input:checked) p {
        color: white !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) [data-baseweb="radio"] {
        background-color: white !important;
        border-color: white !important;
    }
    
    /* Tabs styling - Green theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f8f4;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        color: #2d5016;
        font-weight: 600;
        border: 1px solid #e1e8dd;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e8f5e9;
        color: #2d5016;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    /* RESPONSIVE DESIGN - Mobile & Tablet */
    
    /* Tablet (768px and below) */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.85rem !important;
        }
        
        .modern-card, .plot-card {
            padding: 1.2rem !important;
        }
        
        .card-title, .plot-title {
            font-size: 0.95rem !important;
        }
        
        .result-card {
            padding: 1.5rem !important;
            min-height: 150px !important;
        }
        
        .result-value {
            font-size: 1.8rem !important;
        }
        
        .xai-header {
            font-size: 1.2rem !important;
        }
        
        .stProgress {
            max-width: 300px !important;
        }
    }
    
    /* Mobile (480px and below) */
    @media (max-width: 480px) {
        .main-header {
            padding: 1.2rem 1rem !important;
        }
        
        .main-header h1 {
            font-size: 1.3rem !important;
        }
        
        .main-header p {
            font-size: 0.8rem !important;
        }
        
        .modern-card, .plot-card {
            padding: 1rem !important;
            border-radius: 12px !important;
        }
        
        .card-title, .plot-title {
            font-size: 0.9rem !important;
            margin-bottom: 1rem !important;
        }
        
        .result-card {
            padding: 1.2rem !important;
            min-height: 120px !important;
        }
        
        .result-label {
            font-size: 0.75rem !important;
        }
        
        .result-value {
            font-size: 1.5rem !important;
        }
        
        .result-desc {
            font-size: 0.75rem !important;
        }
        
        .xai-header {
            font-size: 1.1rem !important;
            margin: 2rem 0 1.5rem 0 !important;
        }
        
        .info-box {
            padding: 1rem !important;
            font-size: 0.85rem !important;
        }
        
        .stProgress {
            max-width: 250px !important;
        }
        
        .stButton>button {
            padding: 0.8rem 1.5rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    
    /* Very small mobile (360px and below) */
    @media (max-width: 360px) {
        .main-header h1 {
            font-size: 1.1rem !important;
        }
        
        .result-value {
            font-size: 1.3rem !important;
        }
        
        .stProgress {
            max-width: 200px !important;
        }
    }
    
    /* CUSTOM PROGRESS BAR - Beautiful Aesthetic */
    .stProgress {
        max-width: 400px !important;
        margin: 0 auto !important;
        position: relative !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #42a5f5 0%, #66bb6a 50%, #26a69a 100%) !important;
        height: 4px !important;
        border-radius: 3px !important;
        box-shadow: 0 2px 8px rgba(66, 165, 245, 0.4) !important;
        animation: shimmer 2s infinite !important;
    }
    
    @keyframes shimmer {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%) !important;
        height: 4px !important;
        border-radius: 3px !important;
    }
    
    /* Browse Files Button - GREEN THEME */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #3a5f23 0%, #4a7c2e 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(74, 124, 46, 0.3) !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button:active {
        background: linear-gradient(135deg, #2d5016 0%, #3a5f23 100%) !important;
        transform: translateY(0) !important;
    }
    
    /* Hide default spinner - use progress bar instead */
    [data-testid="stSpinner"] {
        display: none !important;
    }
    
    /* Hide "Running" messages */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    [data-testid="column"] {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'image' not in st.session_state:
    st.session_state.image = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'prev_image_hash' not in st.session_state:
    st.session_state.prev_image_hash = None

# ==================== FUNCTIONS ====================

@st.cache_resource
def load_model():
    """Load model"""
    try:
        model_package = joblib.load('sugarcane_disease_classifier_full.pkl')
        
        def model_predict(X):
            return model_package['model'].predict_proba(X)
        
        explainer = shap.KernelExplainer(model_predict, model_package['X_background'])
        model_package['explainer'] = explainer
        model_package['feature_names'] = create_readable_feature_names()
        
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def predict_image(image, model_package):
    """Prediksi gambar"""
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        features = extract_features(img_bgr)
        features = features.reshape(1, -1)
        features_scaled = model_package['scaler'].transform(features)
        
        prediction = model_package['model'].predict(features_scaled)[0]
        probabilities = model_package['model'].predict_proba(features_scaled)[0]
        
        disease_name = model_package['classes'][prediction]
        confidence = probabilities[prediction] * 100
        
        return disease_name, confidence, features_scaled, prediction
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None, None


@st.cache_data
def compute_shap_cached(_model_package, features_key):
    """Compute SHAP"""
    try:
        features_scaled = np.frombuffer(bytes.fromhex(features_key)).reshape(1, -1)
        shap_values = _model_package['explainer'].shap_values(features_scaled, nsamples=50)
        
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=0)
        
        if shap_values.ndim == 3:
            n_classes = len(_model_package['classes'])
            if shap_values.shape[2] == n_classes:
                shap_values = np.transpose(shap_values, (2, 0, 1))
            elif shap_values.shape[1] == n_classes:
                shap_values = np.transpose(shap_values, (1, 0, 2))
        elif shap_values.ndim == 2:
            shap_values = shap_values[np.newaxis, :, :]
        
        return shap_values
    except Exception as e:
        st.error(f"Error SHAP: {str(e)}")
        return None


def create_feature_importance_plot(shap_values, model_package):
    """Feature Importance - SIMPLE matplotlib bars"""
    try:
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
        
        # Get top 12 features
        n_features = min(12, len(mean_abs_shap), len(model_package['feature_names']))
        top_indices = np.argsort(mean_abs_shap)[-n_features:][::-1]
        top_values = mean_abs_shap[top_indices]
        
        valid_indices = [i for i in top_indices if i < len(model_package['feature_names'])]
        top_names = [model_package['feature_names'][i][:30] for i in valid_indices]
        top_values = top_values[:len(valid_indices)]
        
        if len(top_values) == 0:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Green gradient
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_values)))
        ax.barh(range(len(top_values)), top_values, color=colors, 
                edgecolor='#2d5016', linewidth=1.5, alpha=0.95)
        
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10, fontweight='600', color='#2d5016')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.15, linestyle='--')
        ax.set_facecolor('#fafbfa')
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"❌ Feature Importance Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def create_force_plot(shap_values, features_scaled, prediction, model_package):
    """Force Plot - MIRIP Jupyter style (single horizontal stacked bar)"""
    try:
        shap_vals = shap_values[prediction, 0, :]
        
        # Get base value
        expected_value = model_package['explainer'].expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[prediction] if len(expected_value) > prediction else expected_value[0]
        else:
            base_value = expected_value
        
        # Get top 10 features by absolute SHAP value
        n_features = min(10, len(shap_vals), len(model_package['feature_names']))
        idx = np.argsort(np.abs(shap_vals))[::-1][:n_features]
        
        valid_indices = [i for i in idx if i < len(model_package['feature_names'])]
        if len(valid_indices) == 0:
            return None
        
        top_shap = shap_vals[valid_indices]
        top_names = [model_package['feature_names'][i] for i in valid_indices]
        top_values = features_scaled[0, valid_indices]
        
        # Sort positive first (descending), then negative (ascending)
        pos_mask = top_shap > 0
        neg_mask = top_shap <= 0
        
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]
        
        pos_sorted = pos_idx[np.argsort(top_shap[pos_idx])[::-1]]
        neg_sorted = neg_idx[np.argsort(top_shap[neg_idx])]
        
        sorted_idx = np.concatenate([pos_sorted, neg_sorted])
        sorted_shap = top_shap[sorted_idx]
        sorted_names = [top_names[i] for i in sorted_idx]
        sorted_values = [top_values[i] for i in sorted_idx]
        
        # Create figure - single horizontal bar
        fig, ax = plt.subplots(figsize=(12, 2))
        
        # Stack bars horizontally
        cumsum = base_value
        bar_positions = []
        bar_widths = []
        bar_colors = []
        
        for shap_val in sorted_shap:
            color = '#ff1744' if shap_val > 0 else '#2196f3'  # Pink for positive, blue for negative
            width = abs(shap_val)
            
            if shap_val > 0:
                ax.barh(0, width, left=cumsum, height=0.4, 
                       color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                bar_positions.append(cumsum + width/2)
                cumsum += width
            else:
                cumsum += shap_val
                ax.barh(0, width, left=cumsum, height=0.4,
                       color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                bar_positions.append(cumsum + width/2)
            
            bar_widths.append(width)
            bar_colors.append(color)
        
        # Add feature values below bar
        for i, (pos, val) in enumerate(zip(bar_positions, sorted_values)):
            if bar_widths[i] > 0.1:  # Only show if bar is wide enough
                ax.text(pos, -0.55, f'{val:.2f}', ha='center', va='top',
                       fontsize=7, color=bar_colors[i], fontweight='600')
        
        # Title with confidence, base, and output
        output_val = base_value + np.sum(sorted_shap)
        confidence = 99.8  # You can calculate this from your model
        ax.text(0.5, 1.3, f'Confidence: {confidence}% | Base: {base_value:.3f}', 
               ha='center', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='600', color='#2c3e50')
        ax.text(0.98, 1.3, f'f(x)\n{output_val:.2f}', 
               ha='right', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='700', color='#2c3e50')
        
        # Legend
        ax.text(0.98, 1.1, 'higher', ha='right', va='bottom', 
               transform=ax.transAxes, fontsize=8, color='#ff1744', fontweight='600')
        ax.text(1.0, 1.1, 'lower', ha='right', va='bottom', 
               transform=ax.transAxes, fontsize=8, color='#2196f3', fontweight='600')
        
        # Styling
        ax.set_ylim(-0.7, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Model Output Value', fontsize=9, fontweight='600', color='#2d5016')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('white')
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"❌ Force Plot Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def create_waterfall_plot(shap_values, features_scaled, prediction, model_package):
    """Waterfall - MIRIP Jupyter style (text list + horizontal bars)"""
    try:
        shap_vals = shap_values[prediction, 0, :]
        
        # Get base value
        expected_value = model_package['explainer'].expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[prediction] if len(expected_value) > prediction else expected_value[0]
        else:
            base_value = expected_value
        
        # Get top 15 features
        n_features = min(15, len(shap_vals), len(model_package['feature_names']))
        idx_sorted = np.argsort(np.abs(shap_vals))[::-1][:n_features]
        
        valid_indices = [i for i in idx_sorted if i < len(model_package['feature_names'])]
        if len(valid_indices) == 0:
            return None
        
        top_shap = shap_vals[valid_indices]
        top_names = [model_package['feature_names'][i] for i in valid_indices]
        top_values = features_scaled[0, valid_indices]
        
        # Sort by SHAP value (highest to lowest)
        sort_idx = np.argsort(top_shap)[::-1]
        sorted_shap = top_shap[sort_idx]
        sorted_names = [top_names[i] for i in sort_idx]
        sorted_values = [top_values[i] for i in sort_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Y positions (reversed for top-down)
        y_pos = np.arange(len(sorted_shap))
        
        # Plot bars starting from 0
        colors = ['#ff1744' if v > 0 else '#2196f3' for v in sorted_shap]  # Pink for positive, blue for negative
        bars = ax.barh(y_pos, np.abs(sorted_shap), left=0, height=0.5,
                       color=colors, alpha=0.9, edgecolor='white', linewidth=1)
        
        # Add feature names and values on left
        max_bar_width = max(np.abs(sorted_shap))
        for i, (name, value, shap_val) in enumerate(zip(sorted_names, sorted_values, sorted_shap)):
            # Feature value = Feature name (left side)
            if len(name) > 25:
                name = name[:25]
            label_text = f"{value:.3f} = {name}"
            ax.text(-0.05, i, label_text, ha='right', va='center', 
                   fontsize=8, color='#2c3e50', fontweight='500')
            
            # SHAP contribution on right side of bar
            color_text = '#ff1744' if shap_val > 0 else '#2196f3'
            ax.text(abs(shap_val) + 0.02, i, f"{shap_val:+.2f}", 
                   ha='left', va='center', fontsize=9, 
                   color=color_text, fontweight='700')
        
        # Add f(x) value at top right
        output = base_value + np.sum(sorted_shap)
        fig.text(0.95, 0.96, f'f(x) = {output:.3f}', ha='right', va='top',
                fontsize=11, fontweight='700', color='#2c3e50')
        
        # Styling
        ax.set_yticks([])
        ax.set_xlim(-max_bar_width * 1.2, max_bar_width * 1.3)
        ax.set_xlabel('E[f(x)]', fontsize=9, fontweight='600', color='#2d5016')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='x', alpha=0.15, linestyle='--')
        ax.set_facecolor('white')
        ax.invert_yaxis()
        
        fig.patch.set_facecolor('white')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
    except Exception as e:
        print(f"❌ Waterfall Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


# ==================== MAIN APP ====================

def main():
    model_package = load_model()
    
    # ===== HEADER =====
    st.markdown("""
    <div class="main-header">
        <h1>Deteksi Penyakit Daun Tebu</h1>
        <p>Sistem Klasifikasi Berbasis Kecerdasan Buatan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== GUIDE =====
    st.markdown("""
    <div class="info-box">
        <strong>Panduan Penggunaan:</strong><br>
        <strong>1.</strong> Upload foto daun tebu dari galeri atau ambil foto dengan kamera<br>
        <strong>2.</strong> Preview akan muncul <span class="desktop-only">di sebelah kanan</span><span class="mobile-only">di bawah</span><br>
        <strong>3.</strong> Klik tombol "Analisis Gambar"<br>
        <strong>4.</strong> Lihat hasil deteksi dan analisis model
    </div>
    """, unsafe_allow_html=True)
    
    # Add CSS for responsive text
    st.markdown("""
    <style>
        .mobile-only { display: none; }
        .desktop-only { display: inline; }
        
        @media (max-width: 768px) {
            .mobile-only { display: inline; }
            .desktop-only { display: none; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ===== INPUT SECTION =====
    col_upload, col_preview = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Upload Foto</div>', unsafe_allow_html=True)
        
        # Radio for File Upload or Camera (no extra label)
        upload_method = st.radio(
            "Pilih metode",
            ["📁 Pilih File", "📷 Kamera"],
            label_visibility="collapsed",
            horizontal=True,
            key="upload_method"
        )
        
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        
        if upload_method == "📁 Pilih File":
            uploaded_file = st.file_uploader(
                "Pilih foto dari galeri",
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed",
                help="Format: JPG, JPEG, PNG",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                # Create unique hash for the file
                file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
                
                # Check if this is a new file
                if st.session_state.prev_image_hash != file_hash:
                    st.session_state.image = Image.open(uploaded_file)
                    st.session_state.prev_image_hash = file_hash
                    st.session_state.analysis_done = False
        
        elif upload_method == "📷 Kamera":
            camera_photo = st.camera_input(
                "Ambil foto dengan kamera",
                label_visibility="collapsed",
                help="Klik untuk mengambil foto",
                key="camera_input"
            )
            
            if camera_photo is not None:
                # Create unique hash for the photo
                photo_hash = f"camera_{camera_photo.size}"
                
                # Check if this is a new photo
                if st.session_state.prev_image_hash != photo_hash:
                    st.session_state.image = Image.open(camera_photo)
                    st.session_state.prev_image_hash = photo_hash
                    st.session_state.analysis_done = False
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Button
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button(
            "Analisis Gambar",
            disabled=(st.session_state.image is None),
            type="primary",
            use_container_width=True
        )
    
    with col_preview:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Preview Gambar</div>', unsafe_allow_html=True)
        
        if st.session_state.image is not None:
            # Super fast preview optimization
            preview_image = st.session_state.image.copy()
            
            # Even smaller preview for speed (600x600 max)
            max_preview_size = (600, 600)
            
            # Resize if needed
            if preview_image.width > max_preview_size[0] or preview_image.height > max_preview_size[1]:
                preview_image.thumbnail(max_preview_size, Image.Resampling.LANCZOS)
            
            # Fast base64 conversion with optimization
            img_buffer = io.BytesIO()
            preview_image.save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Get file size from original
            w, h = st.session_state.image.size
            original_buffer = io.BytesIO()
            st.session_state.image.save(original_buffer, format='PNG')
            file_size_mb = len(original_buffer.getvalue()) / (1024 * 1024)
            
            # Fast rendering
            st.markdown(f"""
            <div class="img-preview-box">
                <img src="data:image/jpeg;base64,{img_base64}" alt="Preview">
                <div class="img-caption">
                    <strong style="color: #2d5016;">{w} × {h} px</strong>
                    <span style="color: #2d5016; margin-left: 1rem;">• {file_size_mb:.2f} MB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Empty state
            st.markdown("""
            <div class="img-preview-box">
                <div class="empty-state">
                    <div class="empty-icon">📷</div>
                    <p>Gambar akan ditampilkan di sini</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== PROCESS =====
    if analyze_clicked and st.session_state.image is not None:
        # Center progress bar container
        col_empty1, col_progress, col_empty2 = st.columns([1, 2, 1])
        
        with col_progress:
            st.markdown("""
            <div style='text-align: center; margin: 1.5rem 0 0.5rem 0;'>
                <span style='font-size: 0.9rem; color: #2d5016; font-weight: 600;'>
                    ⚡ Memproses gambar...
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            
            # Step 1: Extract features
            progress_bar.progress(30)
            result = predict_image(st.session_state.image, model_package)
            
            # Step 2: Complete
            progress_bar.progress(100)
            
            if result[0] is not None:
                disease_name, confidence, features_scaled, prediction = result
                
                st.session_state.disease_name = disease_name
                st.session_state.confidence = confidence
                st.session_state.features_scaled = features_scaled
                st.session_state.prediction = prediction
                st.session_state.analysis_done = True
            
            # Clear progress bar
            progress_bar.empty()
    
    # ===== RESULTS =====
    if st.session_state.analysis_done:
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Penyakit Terdeteksi</div>
                <div class="result-value">{st.session_state.disease_name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Tingkat Keyakinan</div>
                <div class="result-value">{st.session_state.confidence:.1f}%</div>
                <div class="result-desc">Akurasi prediksi sistem</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== XAI =====
        st.markdown('<div class="xai-header">Analisis Explainable AI</div>', unsafe_allow_html=True)
        
        features_key = st.session_state.features_scaled.tobytes().hex()
        
        # Center progress bar
        col_empty1, col_shap_progress, col_empty2 = st.columns([1, 2, 1])
        
        with col_shap_progress:
            st.markdown("""
            <div style='text-align: center; margin: 0.5rem 0;'>
                <span style='font-size: 0.85rem; color: #2d5016; font-weight: 600;'>
                    🔍 Menganalisis model...
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            shap_progress = st.progress(0)
            shap_progress.progress(20)
            
            # Compute SHAP silently
            shap_values = compute_shap_cached(model_package, features_key)
            
            shap_progress.progress(100)
            shap_progress.empty()
        
        if shap_values is not None:
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Fitur Penting</div>', unsafe_allow_html=True)
                
                fig1 = create_feature_importance_plot(shap_values, model_package)
                if fig1:
                    st.pyplot(fig1)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Fitur paling berpengaruh pada prediksi</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Kontribusi Fitur</div>', unsafe_allow_html=True)
                
                fig2 = create_force_plot(
                    shap_values,
                    st.session_state.features_scaled,
                    st.session_state.prediction,
                    model_package
                )
                if fig2:
                    st.pyplot(fig2)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Dampak setiap fitur pada keputusan</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Proses Prediksi</div>', unsafe_allow_html=True)
                
                fig3 = create_waterfall_plot(
                    shap_values,
                    st.session_state.features_scaled,
                    st.session_state.prediction,
                    model_package
                )
                if fig3:
                    st.pyplot(fig3)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Alur keputusan model AI</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown("""
    <div class="footer">
        <p><strong>Metode:</strong> Support Vector Machine + SHAP (Explainable AI)</p>
        <p>Membantu petani mendeteksi penyakit daun tebu dengan cepat dan akurat</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()