import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import random

# Kendi modüllerimizi çağırmak için yol ayarı
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import CafaProcessor
from src.model import CafaCNN

def load_model_and_predict():
    # --- 1. AYARLAR ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "cafa_model_v1.pth")
    
    # Cihaz seçimi
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🚀 Tahmin Motoru Başlatılıyor (Cihaz: {device})")

    # --- 2. VERİ İŞLEYİCİYİ HAZIRLA (Sözlüğü Yükle) ---
    # Modelin sayıları anlaması için eğitimdeki aynı haritayı kullanmalıyız.
    print("📖 Sözlük (Etiketler) yükleniyor...")
    processor = CafaProcessor(project_root=project_root, num_labels=1500) # Eğitimdeki sayı ile aynı olmalı
    processor.load_labels() # self.top_terms ve self.term_to_int dolar
    
    # Ters çevrilmiş harita (Index -> GO Term ismi)
    # Örn: 0 -> GO:0005515
    idx_to_term = {v: k for k, v in processor.term_to_int.items()}

    # --- 3. MODELİ YÜKLE ---
    print(f"🧠 Model yükleniyor: {os.path.basename(model_path)}")
    model = CafaCNN(num_labels=1500)
    
    try:
        # Ağırlıkları (weights) dosyadan modele aktar
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Modeli 'Sınav Modu'na al (Eğitimi kapat)
        print("✅ Model hazır!")
    except FileNotFoundError:
        print(f"❌ HATA: Model dosyası bulunamadı! Önce 'train.py' çalıştırılmalı.")
        return

    # --- 4. TAHMİN YAPACAK RASTGELE BİR PROTEİN SEÇ ---
    # Gerçek bir test için FASTA dosyasından rastgele bir tane çekelim
    print("\n🧪 Test için rastgele bir protein seçiliyor...")
    seqs = processor.load_fasta()
    random_id = random.choice(list(seqs.keys()))
    random_seq = seqs[random_id]
    
    print(f"🧬 Protein ID : {random_id}")
    print(f"📏 Uzunluk    : {len(random_seq)} amino asit")
    print(f"📝 Dizi (İlk 50): {random_seq[:50]}...")

    # --- 5. TAHMİN İŞLEMİ (INFERENCE) ---
    # A. Diziyi sayıya çevir
    input_ids = processor.encode_sequence(random_seq)
    # B. Tensöre çevir ve boyut ekle (Batch boyutu: 1) -> [1, 1024]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad(): # Hafızayı yorma, türev alma
        # C. Modele sor
        logits = model(input_tensor)
        # D. Olasılığa çevir (Sigmoid: 0 ile 1 arası yap)
        probs = torch.sigmoid(logits).cpu().numpy()[0] # [0.01, 0.99, 0.05...]

    # --- 6. SONUÇLARI YORUMLA ---
    print("\n🔍 MODELİN TAHMİNLERİ:")
    print("-" * 40)
    
    # En yüksek 5 tahmini bul
    # argsort -> küçükten büyüğe sıralar, [-5:] son 5'i (en büyükleri) alır, [::-1] ters çevirir
    top_5_indices = probs.argsort()[-5:][::-1]
    
    found_any = False
    for idx in top_5_indices:
        score = probs[idx]
        term_id = idx_to_term.get(idx, "Bilinmiyor")
        
        # Sadece %10'un üzerindeki ihtimalleri ciddiye alalım
        if score > 0.01: 
            print(f"🏆 {term_id} : %{score*100:.2f} İhtimal")
            found_any = True
    
    if not found_any:
        print("⚠️ Model bu protein için güçlü bir özellik bulamadı (Düşük güven).")
    
    print("-" * 40)
    print("ℹ️ Not: Bu tahminler, modelin eğitim setindeki 1500 etiketten öğrendikleridir.")

if __name__ == "__main__":
    load_model_and_predict()