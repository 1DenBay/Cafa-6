import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    LEARNING_RATE = 1e-3 # Hatalardan ne kadar hızlı ders çıkarılacak?
    EPOCHS = 5           # Kitap baştan sona kaç kez okunacak?
    NUM_LABELS = 1500    # Kaç etiket tahmin edilecek?

    # --- 2. VERİYİ HAZIRLA (GARSON) ---
    print("\n📊 Veriler Yükleniyor...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = CafaProcessor(project_root=project_root, num_labels=NUM_LABELS)
    
    df_terms = processor.load_labels()
    seqs_dict = processor.load_fasta()
    
    # Ortak ID'leri bul
    common_ids = list(set(df_terms['EntryID']) & set(seqs_dict.keys()))
    print(f"🔗 Eğitim için {len(common_ids)} protein eşleşti.")
    
    # Dataset ve DataLoader
    train_dataset = CafaDataset(processor, common_ids, seqs_dict, df_terms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. MODELİ KUR (BEYİN) ---
    print("\n🧠 Model İnşa Ediliyor...")
    model = CafaCNN(num_labels=NUM_LABELS)
    model.to(device) # Modeli ekran kartına taşı

    # Hakem (Loss) ve Antrenör (Optimizer)
    criterion = nn.BCEWithLogitsLoss() # Çoklu etiket için özel hata ölçer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP) ---
    print(f"\n🔥 Eğitim Başlıyor! ({EPOCHS} Tur)")
    
    for epoch in range(EPOCHS):
        model.train() # Modeli 'Öğrenme Modu'na al
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            # Her 100 pakette bir rapor ver
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Adım [{batch_idx+1}/{len(train_loader)}], Hata: {loss.item():.4f}")
        
        # Bir tur bittiğinde ortalama hatayı yaz
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Tamamlandı! Ortalama Hata: {avg_loss:.4f}")

    # --- 5. KAYDET ---
    save_path = os.path.join(project_root, "models", "cafa_model_v1.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model kaydedildi: {save_path}")

if __name__ == "__main__":
    train()