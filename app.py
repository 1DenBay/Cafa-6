import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from Bio import SeqIO
from io import StringIO
import os

# --- AYARLAR VE SABİTLER ---
st.set_page_config(
    page_title="BioTahmin AI - CNN Modeli",
    page_icon="🧬",
    layout="wide"
)

# Klasör yapına göre model yolu: CAFA-6/models/best_cafa_model.pth -> CNN model kullanılacak
# os.path.join kullanıyoruz ki Windows/Mac fark etmeden yolu bulsun.
MODEL_PATH = os.path.join("models", "best_cafa_model.pth")

# Senin eğitim parametrelerin (Bunlar sabit)
NUM_LABELS = 1500
MAX_LEN = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # sistem neyi destekliyorsa ona göre