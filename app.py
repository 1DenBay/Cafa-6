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

# eğitim parametreleri (Bunlar sabit)
NUM_LABELS = 1500
MAX_LEN = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # sistem neyi destekliyorsa ona göre

# --- 2. MODEL MİMARİSİ (ResNet + LSTM) ---
class ResidualBlock(nn.Module): # modelin gözü
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # modelde kullanılan büyük filtreler (Kernel=9)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class CafaModel(nn.Module): # modelin beyni
    def __init__(self, num_labels):
        super(CafaModel, self).__init__()
        # Harf Sözlüğü: 22 Karakter (20 asit + pad + unknown)
        self.embedding = nn.Embedding(22, 128)
        
        # CNN Katmanları (ResNet) - Özellik çıkarıcı
        self.layer1 = ResidualBlock(128, 256, stride=1)
        self.layer2 = ResidualBlock(256, 512, stride=1)
        
        # LSTM Katmanı (Zaman serisi/Sıralama öğrenir)
        self.lstm = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        
        # Karar Katmanı (Classifier)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, x):
        # 1. Embedding: Harfleri sayısal vektörlere çevir
        x = self.embedding(x).permute(0, 2, 1) # CNN formatına uygun hale getir
        
        # 2. CNN Blokları: Protein üzerindeki desenleri yakala
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 3. LSTM: Sıralamayı ve bağlamı öğren
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # 4. Pooling: En önemli özellikleri seç (Max Pooling)
        x = torch.max(x, dim=1)[0]
        
        # 5. Sonuç: Hangi fonksiyon olduğunu tahmin et
        return self.classifier(x)