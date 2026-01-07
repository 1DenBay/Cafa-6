import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import os
import sys

# Kendi modüllerimizi çağırmak için yol ayarı
# (Bu dosya src içinde olduğu için bir üst dizini görmesi lazım)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import CafaProcessor, CafaDataset
from src.model import CafaCNN

def train():
    # --- 1. AYARLAR VE CİHAZ SEÇİMİ ---
    # Mac (MPS), Nvidia (CUDA) veya İşlemci (CPU) seçimi 
    # bilgisayarın türüne göre en iyi performansı alabilmek için otomatik seçim yapar.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Cihaz: Apple M1/M2/M3 (MPS) - Turbo Modu Aktif!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Cihaz: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("🐢 Cihaz: CPU (Yavaş Mod)")

    # Hiperparametreler (Ayar Düğmeleri)
    BATCH_SIZE = 32      # Her seferde kaç protein incelenecek?
    LEARNING_RATE = 0.0005 # Hatalardan ne kadar hızlı ders çıkarılacak?
    EPOCHS = 8           # Kitap baştan sona kaç kez okunacak?
    NUM_LABELS = 1500    # Kaç etiket tahmin edilecek?
    THRESHOLD = 0.3      # %30'un üzerindeki ihtimalleri "1" kabul et (Kaggle için kritik ayar)

    # --- 2. VERİYİ HAZIRLA (GARSON) ---
    print("\n📊 Veriler Yükleniyor...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = CafaProcessor(project_root=project_root, num_labels=NUM_LABELS)
    
    df_terms = processor.load_labels()
    seqs_dict = processor.load_fasta()
    
    # Ortak ID'leri bul
    all_ids = list(set(df_terms['EntryID']) & set(seqs_dict.keys()))
    print(f"🔗 Toplam Eşleşen Protein: {len(all_ids)}")
    
    # --- KRİTİK ADIM: Train / Validation Ayrımı ---
    # Verinin %20'sini saklıyoruz (Sınav için)
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    print(f"📘 Eğitim Seti   : {len(train_ids)} protein")
    print(f"tc Sınav Seti (Val): {len(val_ids)} protein")

    # Datasetleri oluştur
    train_dataset = CafaDataset(processor, train_ids, seqs_dict, df_terms)
    val_dataset = CafaDataset(processor, val_ids, seqs_dict, df_terms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Sınavda karıştırmaya gerek yok

    # --- 3. MODELİ KUR (BEYİN) ---
    print("\n🧠 Model İnşa Ediliyor...")
    model = CafaCNN(num_labels=NUM_LABELS)
    model.to(device) # Modeli ekran kartına taşı

    # 1'leri bulmak, 0'ları bulmaktan 10 kat daha önemli olsun.
    # Bu sayede model "hepsine 0 basayım" tembelliğinden vazgeçer.
    pos_weight = torch.ones([NUM_LABELS]).to(device) * 10
    # Hakem (Loss) ve Antrenör (Optimizer)
    criterion = nn.BCEWithLogitsLoss(model.parameters(), lr=LEARNING_RATE) # Çoklu etiket için özel hata ölçer
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # En iyi skoru takip etmek için
    best_f1 = 0.0

    # --- 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP) ---
    print(f"\n🔥 Eğitim Başlıyor! ({EPOCHS} Tur)")
    
    for epoch in range(EPOCHS):
        model.train() # Modeli 'Öğrenme Modu'na al
        total_loss = 0
        
        for batch in train_loader:
            # Verileri cihazı taşı (CPU -> GPU/MPS)
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # A. SIFIRLA: Önceki turun artıklarını temizle
            optimizer.zero_grad()
            # B. İLERİ GİT (Forward): Tahmin yap
            outputs = model(inputs)
            # C. HATAYI ÖLÇ (Loss): Ne kadar yanıldık?
            loss = criterion(outputs, labels)
            # D. GERİYE BAK (Backward): Hatanın kaynağını bul
            loss.backward()
            # E. GÜNCELLE (Step): Ağırlıkları düzelt
            optimizer.step()
            total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)

        # B. SINAV (VALIDATION)
        # Dropout kapanır, model sadece bildiğini okur.
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad(): # Hafızayı yorma
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # F1 Hesaplamak için tahminleri al
                # Sigmoid ile 0-1 arasına çekiyoruz
                probs = torch.sigmoid(outputs)
                # Eşik değerinden (0.3) büyükse 1, küçükse 0 yap
                preds = (probs > THRESHOLD).float()
                
                # Listeye ekle (CPU'ya alarak)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Listeleri birleştir
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # C. KARNE (F1 SCORE HESAPLA)
        # 'micro': Genel başarıyı ölçer (Kaggle için iyi bir gösterge)
        val_f1 = f1_score(all_labels, all_preds, average='micro')
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"🏅 F1-Score: {val_f1:.4f}")

        # D. EN İYİYİ KAYDET
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(project_root, "models", "best_cafa_model.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"    💾 Yeni rekor! Model kaydedildi. (Skor: {val_f1:.4f})")