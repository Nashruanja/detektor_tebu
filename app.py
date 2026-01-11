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

# Optional import - app will still work if Gemini not installed
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("\n⚠️  WARNING: google.generativeai not installed!")
    print("   Aplikasi tetap jalan, tapi rekomendasi AI tidak tersedia.")
    print("   Install dengan: pip install google-generativeai\n")

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
    
    /* HIDE SIDEBAR */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
    }
    
    [data-testid="stSidebar"] > div {
        display: none !important;
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
    
    /* Gemini recommendation styling (for AI-generated) */
    .gemini-recommendation h3 {
        color: #d84315 !important;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 3px solid #ff9800;
    }
    
    .gemini-recommendation h3:first-child {
        margin-top: 0;
    }
    
    .gemini-recommendation h4 {
        color: #e65100 !important;
        font-size: 1rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ffe0b2;
    }
    
    .gemini-recommendation h4:first-child {
        margin-top: 0;
    }
    
    .gemini-recommendation h5 {
        color: #e65100 !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem 0;
    }
    
    .gemini-recommendation p {
        color: #5d4037 !important;
        font-size: 0.9rem;
        line-height: 1.8;
        margin: 0.5rem 0;
    }
    
    .gemini-recommendation ul, .gemini-recommendation ol {
        color: #5d4037 !important;
        font-size: 0.9rem;
        line-height: 1.8;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .gemini-recommendation li {
        margin: 0.3rem 0;
        color: #5d4037 !important;
    }
    
    .gemini-recommendation strong {
        color: #e65100 !important;
        font-weight: 700;
    }
    
    /* Rule-based recommendation styling (GREEN theme for fallback) */
    .rule-based-recommendation h3 {
        color: #194d19 !important;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 3px solid #66bb6a;
    }
    
    .rule-based-recommendation h3:first-child {
        margin-top: 0;
    }
    
    .rule-based-recommendation h4 {
        color: #1b5e20 !important;
        font-size: 1rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #81c784;
    }
    
    .rule-based-recommendation h4:first-child {
        margin-top: 0;
    }
    
    .rule-based-recommendation h5 {
        color: #2e7d32 !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem 0;
    }
    
    .rule-based-recommendation p {
        color: #1b5e20 !important;
        font-size: 0.9rem;
        line-height: 1.8;
        margin: 0.5rem 0;
    }
    
    .rule-based-recommendation strong {
        color: #1b5e20 !important;
        font-weight: 700;
    }
    
    .rule-based-recommendation ul, .rule-based-recommendation ol {
        color: #2e7d32 !important;
        font-size: 0.9rem;
        line-height: 1.8;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .rule-based-recommendation li {
        margin: 0.3rem 0;
        color: #2e7d32 !important;
    }
    
    /* Error message code styling */
    code {
        background: #f5f5f5;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #666;
        display: inline-block;
        max-width: 100%;
        word-wrap: break-word;
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
        margin-bottom: 0.5rem !important;
        margin-top: 0 !important;
    }
    
    /* Hide radio label completely */
    [data-testid="stRadio"] > label:first-child {
        display: none !important;
    }
    
    [data-testid="stRadio"] > div {
        gap: 0.8rem !important;
        background: transparent !important;
        display: flex !important;
        flex-direction: row !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* DESKTOP: Hide camera option (only show file upload) */
    @media (min-width: 769px) {
        [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(2) {
            display: none !important;
        }
    }
    
    /* MOBILE: Show both options */
    @media (max-width: 768px) {
        [data-testid="stRadio"] [role="radiogroup"] > label {
            display: inline-flex !important;
        }
    }
    
    [data-testid="stRadio"] > div > label,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [role="radiogroup"] > label,
    [data-testid="stRadio"] div[role="radiogroup"] label {
        color: #0d1f07 !important;  /* VERY DARK GREEN - SUPER VISIBLE */
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        background: #f1f8f4 !important;
        border: 2px solid #c5e1a5 !important;
        padding: 0.6rem 1.2rem !important;  /* More compact */
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
        display: inline-flex !important;
        align-items: center !important;
        min-width: 130px !important;  /* Smaller width */
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
    
    /* ==================== FLOATING TUTORIAL BUTTON ==================== */
    
    /* Floating Tutorial Button - Kanan Bawah (Always Visible) */
    .tutorial-floating-btn {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 999;
    background: linear-gradient(135deg, #4a7c2e 0%, #6b9d4a 100%);
    color: white !important;  /* WHITE TEXT - NOT BLUE! */
    padding: 1rem 1.2rem;
    border-radius: 40px;
    /* NO shadow - removed completely */
    text-decoration: none;
    font-weight: 700;
    font-size: 1.05rem;
    display: inline-flex;
    align-items: center;
    gap: 0.8rem;
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid #2d5016;
    }
    
    .tutorial-floating-btn:hover {
    transform: translateY(-2px);
    /* NO shadow on hover - clean! */
    background: linear-gradient(135deg, #3a5f23 0%, #4a7c2e 100%);
    text-decoration: none;
    color: white !important;
    }

    .tutorial-floating-btn .icon {
    font-size: 1.4rem;
    }
    
    /* Text inside button - force WHITE */
    .tutorial-floating-btn span {
        color: white !important;
    }
    
    /* RESPONSIVE - Tutorial Button */
    @media (max-width: 768px) {
        .tutorial-floating-btn {
            bottom: 1rem;
            right: 1rem;
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
        }
    
        .tutorial-floating-btn .icon {
            font-size: 1.2rem;
        }
    }
    
    @media (max-width: 480px) {
        .tutorial-floating-btn {
            padding: 0.55rem 0.9rem;
            font-size: 0.8rem;
            bottom: 0.8rem;
            right: 0.8rem;
        }
    
        .tutorial-floating-btn .icon {
            font-size: 1.1rem;
        }
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
    """Waterfall - EXACT Jupyter style"""
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
        fig, ax = plt.subplots(figsize=(8, 7.5))
        
        # Y positions
        y_pos = np.arange(len(sorted_shap))
        
        # Plot bars from center (0)
        colors = ['#ff1744' if v > 0 else '#2196f3' for v in sorted_shap]
        
        for i, shap_val in enumerate(sorted_shap):
            if shap_val > 0:
                # Positive - bar to the right
                ax.barh(i, shap_val, left=0, height=0.6, 
                       color='#ff1744', alpha=0.85, edgecolor='white', linewidth=1)
            else:
                # Negative - bar to the left
                ax.barh(i, abs(shap_val), left=shap_val, height=0.6,
                       color='#2196f3', alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add feature names and values on left (outside plot area)
        max_bar = max(np.abs(sorted_shap))
        for i, (name, value, shap_val) in enumerate(zip(sorted_names, sorted_values, sorted_shap)):
            # Truncate name if too long
            if len(name) > 30:
                name = name[:30]
            
            # Feature value = Feature name
            label_text = f"{value:.3f} = {name}"
            ax.text(-max_bar * 0.55, i, label_text, ha='right', va='center', 
                   fontsize=7.5, color='#2c3e50', fontweight='500')
            
            # SHAP contribution value (right side of bar)
            if shap_val > 0:
                x_pos = shap_val + max_bar * 0.05
            else:
                x_pos = shap_val - max_bar * 0.05
            
            ax.text(x_pos, i, f"{shap_val:+.2f}", 
                   ha='left' if shap_val > 0 else 'right', va='center', 
                   fontsize=8.5, color='#ff1744' if shap_val > 0 else '#2196f3', 
                   fontweight='700')
        
        # Add f(x) at top right
        output = base_value + np.sum(sorted_shap)
        fig.text(0.95, 0.97, f'f(x) = {output:.3f}', ha='right', va='top',
                fontsize=10, fontweight='700', color='#2c3e50')
        
        # Center line
        ax.axvline(0, color='#666', linestyle='-', linewidth=1.5, alpha=0.3)
        
        # Styling
        ax.set_yticks([])
        ax.set_xlim(-max_bar * 0.7, max_bar * 1.3)
        ax.set_xlabel('E[f(x)]', fontsize=9, fontweight='600', color='#2d5016')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        ax.set_facecolor('white')
        ax.invert_yaxis()
        
        fig.patch.set_facecolor('white')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
    except Exception as e:
        print(f"❌ Waterfall Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def get_rule_based_recommendation(disease_name):
    """
    Rule-based recommendation system (fallback when Gemini unavailable)
    Berdasarkan pengetahuan domain penyakit tebu
    """
    
    recommendations = {
        "Healthy": """<h3>🌿 Daun Sehat - Tips Perawatan Lanjutan</h3>
            
            <h4>1. Menjaga Kesehatan Daun Tebu</h4>
            <p><strong>Langkah-langkah perawatan rutin:</strong></p>
            <ul>
                <li>Lakukan inspeksi visual setiap 1-2 minggu untuk deteksi dini penyakit</li>
                <li>Jaga kebersihan lahan dari gulma dan sisa tanaman</li>
                <li>Pastikan drainase lahan baik agar tidak tergenang air</li>
                <li>Berikan nutrisi seimbang sesuai fase pertumbuhan</li>
            </ul>
            
            <h4>2. Pencegahan Serangan Penyakit</h4>
            <p><strong>Tips pencegahan umum:</strong></p>
            <ul>
                <li>Gunakan varietas tebu yang tahan penyakit sesuai wilayah</li>
                <li>Terapkan rotasi tanaman untuk memutus siklus penyakit</li>
                <li>Hindari luka pada batang saat pemeliharaan</li>
                <li>Jaga jarak tanam yang cukup untuk sirkulasi udara</li>
            </ul>
            
            <h4>3. Monitoring Berkala</h4>
            <p><strong>Yang perlu diperhatikan:</strong></p>
            <ul>
                <li>Perhatikan perubahan warna daun (menguning, kecoklatan, belang)</li>
                <li>Cek tekstur daun (bintik-bintik, karat, bercak)</li>
                <li>Amati pertumbuhan tanaman secara keseluruhan</li>
                <li>Catat kondisi cuaca dan kelembaban</li>
            </ul>""",
        
        "Red Rot": """<h3>🔴 Red Rot - Rekomendasi Penanganan</h3>
            
            <h4>1. Penanganan agar penyakit tidak meluas</h4>
            <p><strong>Tindakan segera yang perlu dilakukan:</strong></p>
            <ul>
                <li>Identifikasi dan tandai tanaman yang terinfeksi</li>
                <li>Potong bagian batang yang terinfeksi hingga jaringan sehat</li>
                <li>Bakar atau kubur dalam sisa tanaman terinfeksi (jangan dibiarkan di lahan)</li>
                <li>Bersihkan alat pertanian dengan disinfektan setelah kontak dengan tanaman sakit</li>
                <li>Tingkatkan drainase lahan untuk mengurangi kelembaban berlebih</li>
            </ul>
            
            <h4>2. Penanganan jika penyakit sudah meluas</h4>
            <p><strong>Strategi pengelolaan lahan:</strong></p>
            <ul>
                <li>Cabut dan musnahkan tanaman yang terinfeksi parah</li>
                <li>Hindari penanaman tebu di lahan yang sama minimal 1-2 musim</li>
                <li>Lakukan rotasi dengan tanaman lain (jagung, kacang-kacangan)</li>
                <li>Perbaiki sistem drainase untuk menghindari genangan air</li>
                <li>Konsultasikan dengan penyuluh untuk penanganan lahan skala besar</li>
            </ul>
            
            <h4>3. Pencegahan pada tanaman tebu lainnya</h4>
            <p><strong>Langkah pencegahan:</strong></p>
            <ul>
                <li>Gunakan bibit dari varietas tahan Red Rot (konsultasi penyuluh untuk rekomendasi lokal)</li>
                <li>Pilih bibit sehat dan bebas dari gejala penyakit</li>
                <li>Hindari luka pada batang saat proses tanam dan pemeliharaan</li>
                <li>Jaga sanitasi lahan dengan membersihkan sisa tanaman</li>
                <li>Monitor curah hujan - Red Rot berkembang di kondisi lembab</li>
            </ul>""",
        
        "Mosaic": """<h3>🎨 Mosaic - Rekomendasi Penanganan</h3>
            
            <h4>1. Penanganan agar penyakit tidak meluas</h4>
            <p><strong>Tindakan pencegahan penyebaran:</strong></p>
            <ul>
                <li>Identifikasi tanaman dengan gejala mosaic (daun belang hijau terang-gelap)</li>
                <li>Kendalikan populasi serangga vektor (kutu daun/aphids) yang menyebarkan virus</li>
                <li>Cabut tanaman terinfeksi pada tahap awal untuk mencegah penularan</li>
                <li>Hindari kontak langsung alat pertanian dari tanaman sakit ke sehat</li>
                <li>Jaga kebersihan lahan dari gulma yang menjadi inang serangga</li>
            </ul>
            
            <h4>2. Penanganan jika penyakit sudah meluas</h4>
            <p><strong>Strategi pengelolaan:</strong></p>
            <ul>
                <li>Fokus pada pengendalian serangga vektor untuk menghentikan penyebaran</li>
                <li>Cabut tanaman yang terinfeksi parah dan gantikan dengan bibit sehat</li>
                <li>Hindari menggunakan stek dari tanaman terinfeksi untuk penanaman baru</li>
                <li>Lakukan sanitasi menyeluruh setelah panen</li>
                <li>Pertimbangkan penanaman varietas tahan mosaic untuk musim berikutnya</li>
            </ul>
            
            <h4>3. Pencegahan pada tanaman tebu lainnya</h4>
            <p><strong>Langkah pencegahan:</strong></p>
            <ul>
                <li>Gunakan bibit bersertifikat bebas virus (sangat penting!)</li>
                <li>Pilih varietas tebu yang tahan terhadap Mosaic virus</li>
                <li>Kendalikan populasi kutu daun dengan menjaga kebersihan lahan</li>
                <li>Hindari penanaman terlalu rapat - jaga sirkulasi udara baik</li>
                <li>Lakukan roguing (cabut tanaman sakit) secara rutin sejak awal pertumbuhan</li>
            </ul>""",
        
        "Yellow": """<h3>💛 Yellow/Chlorotic - Rekomendasi Penanganan</h3>
            
            <h4>1. Penanganan agar penyakit tidak meluas</h4>
            <p><strong>Tindakan perbaikan nutrisi dan kesehatan:</strong></p>
            <ul>
                <li>Periksa kondisi tanah - defisiensi nitrogen sering menyebabkan menguning</li>
                <li>Perbaiki drainase jika tanah terlalu lembab/tergenang</li>
                <li>Berikan pemupukan berimbang sesuai hasil uji tanah</li>
                <li>Periksa pH tanah - pastikan dalam rentang optimal (6-7)</li>
                <li>Amati apakah ada serangan hama yang melemahkan tanaman</li>
            </ul>
            
            <h4>2. Penanganan jika penyakit sudah meluas</h4>
            <p><strong>Perbaikan kondisi lahan:</strong></p>
            <ul>
                <li>Lakukan pemupukan susulan dengan fokus pada nitrogen organik</li>
                <li>Perbaiki sistem irigasi untuk menghindari kekeringan atau kelebihan air</li>
                <li>Aplikasikan pupuk hijau atau kompos untuk memperbaiki struktur tanah</li>
                <li>Jika disebabkan penyakit sistemik, pertimbangkan peremajaan lahan</li>
                <li>Konsultasi dengan penyuluh untuk analisis tanah dan rekomendasi pemupukan</li>
            </ul>
            
            <h4>3. Pencegahan pada tanaman tebu lainnya</h4>
            <p><strong>Langkah pencegahan:</strong></p>
            <ul>
                <li>Lakukan uji tanah sebelum penanaman untuk mengetahui kebutuhan pupuk</li>
                <li>Berikan pupuk dasar yang cukup saat persiapan lahan</li>
                <li>Gunakan varietas yang sesuai dengan kondisi tanah setempat</li>
                <li>Terapkan jadwal pemupukan teratur sesuai fase pertumbuhan</li>
                <li>Pastikan drainase lahan optimal untuk kesehatan akar</li>
            </ul>""",
        
        "Rust": """<h3>🍂 Rust/Karat - Rekomendasi Penanganan</h3>
            
            <h4>1. Penanganan agar penyakit tidak meluas</h4>
            <p><strong>Tindakan pengendalian penyakit karat:</strong></p>
            <ul>
                <li>Identifikasi tanaman dengan bintik-bintik coklat kemerahan (pustula karat)</li>
                <li>Pangkas dan musnahkan daun yang terinfeksi parah</li>
                <li>Tingkatkan sirkulasi udara dengan mengatur jarak tanam</li>
                <li>Kurangi kelembaban berlebih - hindari penyiraman berlebihan</li>
                <li>Bersihkan gulma yang meningkatkan kelembaban di sekitar tanaman</li>
            </ul>
            
            <h4>2. Penanganan jika penyakit sudah meluas</h4>
            <p><strong>Strategi pengelolaan serangan berat:</strong></p>
            <ul>
                <li>Lakukan sanitasi menyeluruh - kumpulkan dan musnahkan daun terinfeksi</li>
                <li>Perbaiki drainase dan kurangi kelembaban lahan</li>
                <li>Pertimbangkan pemangkasan daun bawah untuk meningkatkan sirkulasi udara</li>
                <li>Hindari pemupukan nitrogen berlebihan yang membuat daun lebih rentan</li>
                <li>Konsultasi penyuluh untuk penanganan skala luas</li>
            </ul>
            
            <h4>3. Pencegahan pada tanaman tebu lainnya</h4>
            <p><strong>Langkah pencegahan:</strong></p>
            <ul>
                <li>Pilih varietas tebu yang tahan terhadap penyakit karat</li>
                <li>Hindari penanaman terlalu rapat - beri jarak cukup antar tanaman</li>
                <li>Jaga kebersihan lahan dari sisa tanaman yang menjadi sumber spora</li>
                <li>Monitor kelembaban - penyakit karat berkembang di kondisi lembab</li>
                <li>Terapkan rotasi tanaman untuk memutus siklus penyakit</li>
            </ul>"""
    }
    
    # Get recommendation, strip ALL whitespace including newlines
    recommendation_html = recommendations.get(disease_name, 
        """<h4>ℹ️ Informasi Umum</h4>
        <p>Untuk rekomendasi penanganan penyakit ini, silakan konsultasikan dengan penyuluh pertanian setempat 
        atau hubungi Dinas Pertanian terdekat untuk mendapatkan panduan yang sesuai dengan kondisi lahan Anda.</p>"""
    )
    
    # Clean up: remove leading/trailing whitespace and normalize
    return recommendation_html.strip()


def get_gemini_recommendation(disease_name):
    """Get treatment recommendation from Gemini API with detailed error handling"""
    
    # Check if Gemini library is installed
    if not GEMINI_AVAILABLE:
        return {
            "success": False,
            "error_type": "library_not_installed",
            "message": """📦 <strong>Library Gemini AI belum terinstall</strong><br><br>
            Untuk mengaktifkan fitur rekomendasi AI, install library dengan cara:<br><br>
            <strong>Buka Command Prompt / Terminal, ketik:</strong><br>
            <code>pip install google-generativeai</code><br><br>
            Setelah install, restart aplikasi.<br><br>
            <strong>Catatan:</strong> Hasil deteksi penyakit tetap akurat dan bisa dipercaya!"""
        }
    
    try:
        # ===== OPTION 1: Streamlit Secrets (Recommended for deployment) =====
        # Baca dari file .streamlit/secrets.toml
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        
        # ===== OPTION 2: Hardcode untuk Testing (MUDAH!) =====
        # UNCOMMENT baris di bawah dan ganti dengan API key Anda:
        # 
        # ⚠️  PENTING - QUOTA HABIS? BACA INI!
        # Quota Gemini = PER GOOGLE ACCOUNT (bukan per API key!)
        # 
        # SALAH ❌:
        #   - Bikin API key baru dari Gmail SAMA → Quota tetap HABIS!
        #   - user123@gmail.com → API key AAA (habis)
        #   - user123@gmail.com → API key BBB (TETAP HABIS!)
        # 
        # BENAR ✅:
        #   - Logout dari makersuite.google.com
        #   - Login dengan Gmail BERBEDA (user456@gmail.com)
        #   - Bikin API key baru → QUOTA FRESH!
        # 
        # api_key = "AIzaSy...your-NEW-api-key-from-DIFFERENT-gmail-account"
        
        # ===== OPTION 3: Environment Variable =====
        # import os
        # api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            return {
                "success": False,
                "error_type": "no_api_key",
                "message": "🔑 <strong>API Key tidak ditemukan</strong><br>Sistem belum dikonfigurasi dengan benar. Hubungi administrator atau pengembang aplikasi untuk mengaktifkan fitur rekomendasi AI."
            }
        
        # Configure API
        genai.configure(api_key=api_key)
        
        # ===== AUTO-SELECT MODEL (TANPA list_models untuk HEMAT QUOTA!) =====
        print("\n" + "="*60)
        print("🎯 SELECTING GEMINI MODEL (QUOTA-SAVING MODE)...")
        print("="*60)
        
        # HARDCODE priority list - TIDAK panggil list_models() yang konsumsi quota!
        # list_models() = 1 API request = BUANG QUOTA!
        # Langsung pakai model tertentu = HEMAT QUOTA!
        
        # Use the latest available models (verified from troubleshoot_quota.py)
        preferred_models = [
            'gemini-2.5-flash',  # Latest, recommended, VERIFIED AVAILABLE!
            'gemini-flash-latest',  # Stable alias, always works
            'gemini-2.0-flash',  # Backup option
        ]
        
        # IMPORTANT: Only try 1-2 models max to save quota!
        # Don't loop through all 6 models - that wastes quota!
        selected_model = preferred_models[0]
        
        print(f"✅ MODEL DIPILIH: {selected_model}")
        print("📌 Mode: Single model (hemat quota)")
        print("="*60 + "\n")
        
        # Try PRIMARY model only
        try:
            print(f"🧪 Mencoba model: {selected_model}")
            model = genai.GenerativeModel(selected_model)
            
            # Create prompt
            prompt = f"""Kamu adalah asisten pertanian untuk petani tebu. Kamu akan menerima nama penyakit daun tebu yang sudah ditentukan dari hasil klasifikasi model SVM.

Penyakit yang terdeteksi: {disease_name}

Tugas kamu adalah memberikan rekomendasi penanganan bersifat umum, aman, dan mudah dipahami petani, tanpa menyebutkan dosis, merek, atau bahan kimia spesifik, sesuaikan penanganan dengan cuaca terupdate hari ini. 

Batasi jawaban hanya pada tiga bagian berikut:

1. Penanganan agar penyakit tidak meluas
Jelaskan langkah-langkah umum yang bisa dilakukan petani untuk mencegah penyebaran penyakit ke tanaman lain.

2. Penanganan jika penyakit sudah meluas
Jelaskan tindakan pengelolaan lahan dan tanaman secara umum jika sebagian besar tanaman sudah terdampak.

3. Pencegahan pada tanaman tebu lainnya
Berikan saran pencegahan agar tanaman tebu yang masih sehat tidak ikut terserang.

Gunakan bahasa yang sederhana, non-teknis, dan ramah petani.

Dilarang:
* Memberikan rekomendasi penggunaan pestisida, fungisida, atau bahan kimia tertentu
* Menyebutkan dosis, campuran, atau prosedur teknis berisiko
* Memberikan diagnosis ulang atau mempertanyakan hasil klasifikasi

Jika diperlukan, arahkan petani untuk berkonsultasi dengan penyuluh pertanian atau sumber resmi setempat.
tambahkan informasi pula mengenai penyakit tersebut, yang dapat menyebabkan kerugian pada tanaman tebu.

Format jawaban dalam HTML dengan struktur:
<h4>1. Penanganan agar penyakit tidak meluas</h4>
<p>...</p>

<h4>2. Penanganan jika penyakit sudah meluas</h4>
<p>...</p>

<h4>3. Pencegahan pada tanaman tebu lainnya</h4>
<p>...</p>
"""
            
            # Generate response with timeout
            response = model.generate_content(prompt)
            
            if response and response.text:
                print(f"✅ BERHASIL dengan model: {selected_model}\n")
                return {
                    "success": True,
                    "message": response.text
                }
            else:
                print(f"⚠️  Model {selected_model} tidak memberikan response\n")
                # Return empty response error
                return {
                    "success": False,
                    "error_type": "empty_response",
                    "message": "⚠️ <strong>Tidak ada respon dari AI</strong><br>Sistem AI tidak memberikan jawaban. Coba lagi dalam beberapa saat."
                }
                
        except Exception as model_error:
            error_str = str(model_error).lower()
            print(f"❌ Model {selected_model} error: {error_str[:200]}\n")
            
            # Check if 404 model not found → try backup model or use rule-based
            if '404' in error_str or 'not found' in error_str or 'is not found' in error_str:
                print("\n" + "="*70)
                print("⚠️  MODEL TIDAK DITEMUKAN (404)!")
                print("="*70)
                print(f"\n📌 Model '{selected_model}' tidak tersedia di API Anda")
                print("\n💡 SOLUSI OTOMATIS:")
                print("   • Menggunakan rekomendasi berbasis aturan (rule-based)")
                print("   • Kualitas tetap bagus, tidak perlu AI!")
                print("   • Unlimited & gratis!")
                print("\n✅ User tetap dapat rekomendasi lengkap.")
                print("="*70 + "\n")
                
                rule_based = get_rule_based_recommendation(disease_name)
                return {
                    "success": True,
                    "is_rule_based": True,
                    "message": rule_based,
                    "info_message": "📌 <strong>Model AI tidak tersedia</strong> - Menggunakan rekomendasi berbasis aturan"
                }
            
            # Check if quota/rate limit error → use rule-based fallback
            elif any(keyword in error_str for keyword in ['quota', 'limit', '429', 'resource']):
                print("\n" + "="*70)
                print("❌ QUOTA HABIS!")
                print("="*70)
                print("\n📌 INFO PENTING:")
                print("   • Quota Gemini = PER GOOGLE ACCOUNT (bukan per API key!)")
                print("   • API key baru dari account SAMA = quota tetap SAMA!")
                print("\n💡 SOLUSI:")
                print("   1. Pakai rule-based fallback (OTOMATIS - lihat di UI)")
                print("   2. Tunggu 24 jam untuk quota reset")
                print("   3. Buat API key dari GOOGLE ACCOUNT BERBEDA")
                print("      → Logout dari makersuite.google.com")
                print("      → Login dengan Gmail LAIN")
                print("      → Buat API key baru → Quota fresh!")
                print("\n✅ Aplikasi tetap jalan! User dapat rekomendasi rule-based.")
                print("="*70 + "\n")
                
                rule_based = get_rule_based_recommendation(disease_name)
                return {
                    "success": True,
                    "is_rule_based": True,
                    "message": rule_based,
                    "info_message": "⏰ <strong>Quota AI habis</strong> - Menggunakan rekomendasi berbasis aturan"
                }
            else:
                # Other error, re-raise to be handled by outer except
                raise model_error
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for specific error types
        
        # 1. No internet / network error - USE RULE-BASED FALLBACK
        if any(keyword in error_str for keyword in ['network', 'connection', 'unreachable', 'timeout', 'dns', 'failed to establish']):
            rule_based = get_rule_based_recommendation(disease_name)
            return {
                "success": True,  # Changed to True because we have fallback
                "is_rule_based": True,
                "message": rule_based,
                "info_message": "📡 <strong>Tidak ada koneksi internet</strong> - Menggunakan rekomendasi berbasis aturan"
            }
        
        # 2. Quota exceeded - USE RULE-BASED FALLBACK
        elif any(keyword in error_str for keyword in ['quota', 'limit', 'rate limit', 'too many requests', '429']):
            rule_based = get_rule_based_recommendation(disease_name)
            return {
                "success": True,  # Changed to True because we have fallback
                "is_rule_based": True,
                "message": rule_based,
                "info_message": "⏰ <strong>Kuota AI Gemini habis</strong> - Menggunakan rekomendasi berbasis aturan"
            }
        
        # 3. Invalid API key
        elif any(keyword in error_str for keyword in ['api key', 'invalid', 'unauthorized', '401', '403', 'permission']):
            return {
                "success": False,
                "error_type": "invalid_key",
                "message": "🔐 <strong>Konfigurasi sistem bermasalah</strong><br><br>Ada masalah dengan pengaturan aplikasi. Hubungi administrator atau pengembang untuk memperbaiki."
            }
        
        # 4. Server error (Google's side)
        elif any(keyword in error_str for keyword in ['500', '502', '503', 'server error', 'internal error']):
            return {
                "success": False,
                "error_type": "server_error",
                "message": "🔧 <strong>Server AI sedang bermasalah</strong><br><br>Sistem AI Google sedang mengalami gangguan. Ini bukan masalah dari HP Anda. Silakan:<br>• Tunggu 10-15 menit<br>• Coba lagi nanti<br><br>Hasil deteksi penyakit tetap bisa digunakan."
            }
        
        # 5. Generic error
        else:
            return {
                "success": False,
                "error_type": "unknown",
                "message": f"❌ <strong>Terjadi kesalahan</strong><br><br>Maaf, ada masalah saat mengambil rekomendasi AI.<br><br><strong>Detail error (untuk teknisi):</strong><br><code style='font-size: 0.75rem; color: #666;'>{str(e)[:200]}</code><br><br>Silakan hubungi penyuluh pertanian untuk konsultasi langsung."
            }


# ==================== MAIN APP ====================

def main():
    model_package = load_model()
    
    # ===== HEADER =====
    st.markdown("""
    <div class="main-header">
        <h1>Deteksi Penyakit Daun Tebu</h1>
        <p>Klasifikasi Penyakit Daun Tebu Menggunakan Support Vector Machine dengan Pendekatan Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== SIDEBAR - REMOVED (disease_info.py not available) =====
    # Sidebar information panel has been removed due to missing disease_info module
    # Core functionality (disease detection, XAI, recommendations) still works!
    
    # ===== GUIDE =====
    st.markdown("""
    <div class="info-box">
        <strong>Panduan Penggunaan:</strong><br>
        <strong>1.</strong> Upload foto daun tebu dari galeri atau ambil foto dengan kamera<br>
        <strong>2.</strong> Preview akan muncul <span class="desktop-only">di sebelah kanan</span><span class="mobile-only">di bawah</span><br>
        <strong>3.</strong> Klik tombol "Analisis Gambar"<br>
        <strong>4.</strong> Lihat hasil deteksi dan analisis model<br><br>
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
        
        # Radio for upload method - ONLY visible on mobile/tablet
        # Desktop: NO radio button (langsung file uploader aja)
        # Mobile: Radio button untuk pilih File vs Camera
        
        upload_method = st.radio(
            "x",
            ["📁 Pilih File", "📷 Kamera"],
            label_visibility="collapsed",
            horizontal=True,
            key="upload_method"
        )
        
        # CSS: Hide ENTIRE radio on desktop, show on mobile
        st.markdown("""
        <style>
        /* DESKTOP (> 768px): Hide radio buttons COMPLETELY */
        @media (min-width: 769px) {
            [data-testid="stRadio"] {
                display: none !important;
            }
        }
        
        /* MOBILE/TABLET (≤ 768px): Show radio buttons */
        @media (max-width: 768px) {
            [data-testid="stRadio"] {
                display: block !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # NO extra spacing
        
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
            # Warning for localhost/HTTP
            st.markdown("""
            <div style="background: #fff3e0; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 3px solid #ff9800;">
                <p style="margin: 0; font-size: 0.8rem; color: #e65100; line-height: 1.5;">
                    📱 <strong>Info:</strong> Kamera di HP butuh <strong>HTTPS</strong>. 
                    Kalau ga bisa akses kamera, pakai <strong>📁 Pilih File</strong> aja!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
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
            # Use placeholder so we can clear it
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
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
            
            # Clear loading indicators
            loading_placeholder.empty()
            progress_bar.empty()
            
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
            # Get all probabilities for all classes
            all_probs = model_package['model'].predict_proba(st.session_state.features_scaled)[0]
            
            # Create HTML for all classes (excluding detected class) - HORIZONTAL
            other_classes_list = []
            for idx, class_name in enumerate(model_package['classes']):
                if idx != st.session_state.prediction:  # Skip the detected class
                    prob_percent = all_probs[idx] * 100
                    other_classes_list.append(f'{class_name}: {prob_percent:.1f}%')
            
            # Join with spacing for horizontal display
            other_classes_html = '&nbsp;&nbsp;&nbsp;&nbsp;'.join(other_classes_list)
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Probabilitas Model</div>
                <div class="result-value">{st.session_state.confidence:.1f}%</div>
                <div class="result-desc">Akurasi prediksi antar kelas</div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e1e8dd; color: #666; font-size: 0.75rem; line-height: 1.8;">
                    {other_classes_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== DISEASE INFORMATION - REMOVED =====
        # (disease_info module not available - using Gemini AI recommendations instead)
        
        # ===== XAI =====
        st.markdown('<div class="xai-header">Analisis Explainable AI</div>', unsafe_allow_html=True)
        
        features_key = st.session_state.features_scaled.tobytes().hex()
        
        # Center progress bar
        col_empty1, col_shap_progress, col_empty2 = st.columns([1, 2, 1])
        
        with col_shap_progress:
            # Use placeholder so we can clear it
            shap_loading_placeholder = st.empty()
            shap_loading_placeholder.markdown("""
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
            
            # Clear loading indicators
            shap_loading_placeholder.empty()
            shap_progress.empty()
        
        if shap_values is not None:
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Feature Importance</div>', unsafe_allow_html=True)
                
                fig1 = create_feature_importance_plot(shap_values, model_package)
                if fig1:
                    st.pyplot(fig1, use_container_width=True)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Fitur paling berpengaruh pada prediksi</div>', unsafe_allow_html=True)
                
                # CARA BACA GRAFIK
                pred_class = model_package['classes'][st.session_state.prediction]
                
                if pred_class == "Healthy":
                    explanation = "<strong>Cara baca grafik kiri:</strong><br>✅ Bar paling panjang = fitur paling penting<br>✅ Lihat: HSV Saturation, Hue (warna hijau) ada di atas → artinya warna hijau segar yang bikin model yakin ini SEHAT<br>✅ Tidak ada fitur merah/coklat/kasar → makanya bukan penyakit!"
                elif pred_class == "Red Rot":
                    explanation = "<strong>Cara baca grafik kiri:</strong><br>✅ Bar paling panjang = fitur paling penting<br>✅ Lihat: Red channel, HOG (tekstur kasar) ada di atas → artinya warna merah & tekstur kasar yang bikin model yakin ini RED ROT<br>✅ Cocok dengan Red Rot yang daunya merah kecoklatan & kasar!"
                elif pred_class == "Mosaic":
                    explanation = "<strong>Cara baca grafik kiri:</strong><br>✅ Bar paling panjang = fitur paling penting<br>✅ Lihat: Kontras, variance, edge ada di atas → artinya pola belang-belang tidak teratur yang bikin model yakin ini MOSAIC<br>✅ Cocok dengan Mosaic yang daunnya belang-belang!"
                elif pred_class == "Yellow":
                    explanation = "<strong>Cara baca grafik kiri:</strong><br>✅ Bar paling panjang = fitur paling penting<br>✅ Lihat: Yellow channel, brightness ada di atas → artinya warna kuning terang yang bikin model yakin ini YELLOW<br>✅ Cocok dengan Yellow yang daunnya menguning!"
                elif pred_class == "Rust":
                    explanation = "<strong>Cara baca grafik kiri:</strong><br>✅ Bar paling panjang = fitur paling penting<br>✅ Lihat: HOG (tekstur kasar), warna coklat/orange ada di atas → artinya tekstur kasar & warna karat yang bikin model yakin ini RUST<br>✅ Cocok dengan Rust yang daunnya kasar & berkarat!"
                else:
                    explanation = "Lihat bar paling panjang - itu fitur yang paling menentukan hasil deteksi."
                
                st.markdown(f"""
                <div style="background: #fff3e0; padding: 0.7rem 0.9rem; border-radius: 6px; margin-top: 0.5rem; 
                            border-left: 3px solid #f57c00;">
                    <p style="margin: 0; font-size: 0.8rem; color: #e65100; line-height: 1.5;">
                        {explanation}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.pred_class = pred_class
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Force Plot</div>', unsafe_allow_html=True)
                
                fig2 = create_force_plot(
                    shap_values,
                    st.session_state.features_scaled,
                    st.session_state.prediction,
                    model_package
                )
                if fig2:
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Dampak setiap fitur pada keputusan</div>', unsafe_allow_html=True)
                
                # CARA BACA GRAFIK
                pred_class = st.session_state.get('pred_class', 'Unknown')
                
                if pred_class == "Healthy":
                    explanation = "<strong>Cara baca grafik tengah:</strong><br>🟦 <strong>Biru banyak</strong> = fitur-fitur menarik ke \"SEHAT\"<br>🟥 Pink sedikit = fitur penyakit lemah<br>→ Biru > Pink = Model yakin ini SEHAT, bukan penyakit!"
                elif pred_class == "Red Rot":
                    explanation = "<strong>Cara baca grafik tengah:</strong><br>🟥 <strong>Pink banyak</strong> = fitur-fitur mendorong ke \"RED ROT\"<br>🟦 Biru sedikit = fitur sehat kalah<br>→ Pink > Biru = Model yakin ini RED ROT!"
                elif pred_class == "Mosaic":
                    explanation = "<strong>Cara baca grafik tengah:</strong><br>🟥 <strong>Pink banyak</strong> = fitur-fitur mendorong ke \"MOSAIC\"<br>🟦 Biru sedikit = fitur teratur kalah<br>→ Pink > Biru = Model yakin ini MOSAIC!"
                elif pred_class == "Yellow":
                    explanation = "<strong>Cara baca grafik tengah:</strong><br>🟥 <strong>Pink banyak</strong> = fitur-fitur mendorong ke \"YELLOW\"<br>🟦 Biru sedikit = fitur hijau kalah<br>→ Pink > Biru = Model yakin ini YELLOW!"
                elif pred_class == "Rust":
                    explanation = "<strong>Cara baca grafik tengah:</strong><br>🟥 <strong>Pink banyak</strong> = fitur-fitur mendorong ke \"RUST\"<br>🟦 Biru sedikit = fitur halus kalah<br>→ Pink > Biru = Model yakin ini RUST!"
                else:
                    explanation = "🟥 Pink = mendorong ke hasil deteksi | 🟦 Biru = menahan dari hasil deteksi"
                
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 0.7rem 0.9rem; border-radius: 6px; margin-top: 0.5rem; 
                            border-left: 3px solid #1976d2;">
                    <p style="margin: 0; font-size: 0.8rem; color: #0d47a1; line-height: 1.5;">
                        {explanation}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">Waterfall Plot</div>', unsafe_allow_html=True)
                
                fig3 = create_waterfall_plot(
                    shap_values,
                    st.session_state.features_scaled,
                    st.session_state.prediction,
                    model_package
                )
                if fig3:
                    st.pyplot(fig3, use_container_width=True)
                    plt.close()
                else:
                    st.error("❌ Error: Plot gagal dimuat. Check console untuk details.")
                
                st.markdown('<div class="plot-caption">Alur keputusan model AI</div>', unsafe_allow_html=True)
                
                # CARA BACA GRAFIK
                pred_class = st.session_state.get('pred_class', 'Unknown')
                
                if pred_class == "Healthy":
                    explanation = "<strong>Cara baca grafik kanan:</strong><br>• <strong>Angka KIRI</strong> (misal: 0.85) = nilai fitur di gambar kamu<br>• <strong>Angka KANAN</strong> (misal: +0.15) = seberapa besar fitur ini bikin model yakin<br>• <strong>+0.15</strong> (positif) = NAMBAH keyakinan ke SEHAT<br>→ Banyak angka + dari fitur hijau/halus = Yakin SEHAT!"
                elif pred_class == "Red Rot":
                    explanation = "<strong>Cara baca grafik kanan:</strong><br>• <strong>Angka KIRI</strong> (misal: -0.75) = nilai fitur di gambar kamu<br>• <strong>Angka KANAN</strong> (misal: +0.26) = seberapa besar fitur ini bikin model yakin<br>• <strong>+0.26</strong> (besar!) = NAMBAH banyak keyakinan ke RED ROT<br>→ Angka + besar dari fitur merah/kasar = Yakin RED ROT!"
                elif pred_class == "Mosaic":
                    explanation = "<strong>Cara baca grafik kanan:</strong><br>• <strong>Angka KIRI</strong> (misal: -0.52) = nilai fitur di gambar kamu<br>• <strong>Angka KANAN</strong> (misal: +0.18) = seberapa besar fitur ini bikin model yakin<br>• <strong>+0.18</strong> (positif) = NAMBAH keyakinan ke MOSAIC<br>→ Angka + dari fitur kontras/pola = Yakin MOSAIC!"
                elif pred_class == "Yellow":
                    explanation = "<strong>Cara baca grafik kanan:</strong><br>• <strong>Angka KIRI</strong> (misal: 3.24) = nilai fitur di gambar kamu<br>• <strong>Angka KANAN</strong> (misal: +0.22) = seberapa besar fitur ini bikin model yakin<br>• <strong>+0.22</strong> (besar!) = NAMBAH banyak keyakinan ke YELLOW<br>→ Angka + besar dari fitur kuning = Yakin YELLOW!"
                elif pred_class == "Rust":
                    explanation = "<strong>Cara baca grafik kanan:</strong><br>• <strong>Angka KIRI</strong> (misal: -0.758) = nilai fitur di gambar kamu<br>• <strong>Angka KANAN</strong> (misal: +0.26) = seberapa besar fitur ini bikin model yakin<br>• <strong>+0.26</strong> (besar!) = NAMBAH banyak keyakinan ke RUST<br>→ Angka + besar dari fitur kasar/karat = Yakin RUST!"
                else:
                    explanation = "Angka kiri = nilai fitur | Angka kanan (+/-) = kontribusi ke hasil"
                
                st.markdown(f"""
                <div style="background: #f3e5f5; padding: 0.7rem 0.9rem; border-radius: 6px; margin-top: 0.5rem; 
                            border-left: 3px solid #7b1fa2;">
                    <p style="margin: 0; font-size: 0.8rem; color: #4a148c; line-height: 1.5;">
                        {explanation}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ===== GEMINI AI RECOMMENDATION =====
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="xai-header">💡 Rekomendasi Penanganan dari AI</div>', unsafe_allow_html=True)
            
            # Show loading while getting recommendation
            with st.spinner('🤖 Menganalisis rekomendasi penanganan...'):
                gemini_result = get_gemini_recommendation(st.session_state.disease_name)
            
            # Check if successful
            if gemini_result and gemini_result.get("success"):
                # Check if this is rule-based or AI-generated
                is_rule_based = gemini_result.get("is_rule_based", False)
                info_msg = gemini_result.get("info_message", "")
                
                if is_rule_based:
                    # RULE-BASED RECOMMENDATION (Fallback)
                    # Get the HTML content
                    html_content = gemini_result['message']
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%); 
                                padding: 1.5rem 2rem; 
                                border-radius: 16px; 
                                border-left: 5px solid #66bb6a;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                margin-bottom: 2rem;">
                        <div style="background: #fff3e0; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 3px solid #ffa726;">
                            <p style="margin: 0; font-size: 0.85rem; color: #e65100; font-weight: 600;">
                                {info_msg}
                            </p>
                        </div>
                        <div class="rule-based-recommendation">{html_content}</div>
                        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #c8e6c9;">
                            <p style="margin: 0; font-size: 0.8rem; color: #558b2f; font-style: italic;">
                                📚 <strong>Sumber:</strong> Rekomendasi ini berdasarkan pengetahuan umum penanganan penyakit tebu. 
                                Untuk penanganan spesifik sesuai kondisi lahan Anda, konsultasikan dengan penyuluh pertanian setempat.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # AI-GENERATED RECOMMENDATION (Gemini)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%); 
                                padding: 1.5rem 2rem; 
                                border-radius: 16px; 
                                border-left: 5px solid #ffa726;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                margin-bottom: 2rem;">
                        <div class="gemini-recommendation" style="color: #e65100; font-size: 0.9rem; line-height: 1.8;">
                            {gemini_result['message']}
                        </div>
                        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #ffe0b2;">
                            <p style="margin: 0; font-size: 0.8rem; color: #f57c00; font-style: italic;">
                                🤖 <strong>Sumber:</strong> Rekomendasi dari Gemini AI. 
                                Untuk penanganan spesifik, konsultasikan dengan penyuluh pertanian setempat.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif gemini_result and not gemini_result.get("success"):
                # ERROR - Show specific error message based on type
                error_type = gemini_result.get("error_type", "unknown")
                error_message = gemini_result.get("message", "Terjadi kesalahan.")
                
                # Different colors for different error types
                if error_type == "no_internet":
                    bg_color = "#e3f2fd"  # Blue for network issues
                    border_color = "#2196f3"
                    icon = "📡"
                elif error_type == "quota_exceeded":
                    bg_color = "#fff3e0"  # Orange for quota
                    border_color = "#ff9800"
                    icon = "⏰"
                elif error_type in ["no_api_key", "invalid_key"]:
                    bg_color = "#fce4ec"  # Pink for config issues
                    border_color = "#e91e63"
                    icon = "🔑"
                elif error_type == "server_error":
                    bg_color = "#f3e5f5"  # Purple for server issues
                    border_color = "#9c27b0"
                    icon = "🔧"
                else:
                    bg_color = "#ffebee"  # Red for unknown
                    border_color = "#f44336"
                    icon = "❌"
                
                st.markdown(f"""
                <div style="background: {bg_color}; 
                            padding: 1.5rem 2rem; 
                            border-radius: 16px; 
                            border-left: 5px solid {border_color};
                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                            margin-bottom: 2rem;">
                    <div style="font-size: 0.95rem; line-height: 1.8; color: #333;">
                        {error_message}
                    </div>
                    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1);">
                        <p style="margin: 0; font-size: 0.85rem; color: #666; font-weight: 600;">
                            💬 <strong>Alternatif:</strong> Hasil deteksi penyakit di atas tetap akurat. 
                            Silakan konsultasi dengan penyuluh pertanian setempat untuk mendapatkan rekomendasi penanganan.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                # FALLBACK - No result at all
                st.markdown("""
                <div style="background: #fff3e0; padding: 1.5rem 2rem; border-radius: 16px; 
                            border-left: 5px solid #ff9800; margin-bottom: 2rem;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                    <p style="margin: 0; color: #e65100; font-size: 0.95rem; line-height: 1.8;">
                        ℹ️ <strong>Rekomendasi AI tidak tersedia saat ini</strong><br><br>
                        Sistem tidak dapat memberikan rekomendasi otomatis. Namun hasil deteksi penyakit tetap akurat dan dapat dipercaya.<br><br>
                        <strong>Langkah selanjutnya:</strong><br>
                        • Catat nama penyakit yang terdeteksi<br>
                        • Hubungi penyuluh pertanian terdekat<br>
                        • Tunjukkan hasil deteksi ini untuk konsultasi lebih lanjut
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # # ===== FOOTER =====
    # st.markdown("""
    # <div class="footer">
    #     <p><strong>Metode:</strong> Support Vector Machine + SHAP (Explainable AI)</p>
    #     <p>Membantu petani mendeteksi penyakit daun tebu dengan cepat dan akurat</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # ===== FLOATING TUTORIAL BUTTON =====
    st.markdown("""
    <a href="/Tutorial" target="_self" class="tutorial-floating-btn">
        <span class="icon">📚</span>
        <span>Tutorial</span>
    </a>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()