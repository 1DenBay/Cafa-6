import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm

# Yolları ayarla
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import CafaProcessor
from src.model import CafaCNN

def make_submission():
    # --- 1. AYARLAR ---
    BATCH_SIZE = 32      
    NUM_LABELS = 1500
    THRESHOLD = 0.01     # Kaggle için düşük bir eşik değeri
    
    # Cihaz ayarı
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Cihaz: NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Cihaz: Apple MPS")
    else:
        device = torch.device("cpu")
        print("🐢 Cihaz: CPU")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- AKILLI MODEL SEÇİMİ ---
    # Önce Drive'a bak, yoksa yerel klasöre bak
    drive_model_path = "/content/drive/MyDrive/Cafa_Models/best_cafa_model.pth"
    local_model_path = os.path.join(project_root, "models", "best_cafa_model.pth")
    
    if os.path.exists(drive_model_path):
        model_path = drive_model_path
        print(f"📂 Model Drive'dan yükleniyor: {model_path}")
    elif os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"📂 Model yerel diskten yükleniyor: {model_path}")
    else:
        print(f"❌ HATA: Model dosyası bulunamadı! (Drive'da veya models klasöründe yok)")
        return

    test_fasta_path = os.path.join(project_root, "data", "raw", "Test", "test_sequences.fasta")
    submission_path = os.path.join(project_root, "submission.tsv")

    # --- 2. HAZIRLIK ---
    print("📖 Sözlük ve Test Verisi yükleniyor...")
    processor = CafaProcessor(project_root=project_root, num_labels=NUM_LABELS)
    processor.load_labels() 
    
    int_to_term = {v: k for k, v in processor.term_to_int.items()}

    processor.fasta_path = test_fasta_path
    test_seqs = processor.load_fasta()
    test_ids = list(test_seqs.keys())
    print(f"🔗 Toplam Test Proteini: {len(test_ids)}")

    # --- 3. MODELİ YÜKLE ---
    print(f"🧠 Model yükleniyor...")
    model = CafaCNN(num_labels=NUM_LABELS)
    
    # Modeli yükle
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 4. TAHMİN DÖNGÜSÜ ---
    print("⚡ Tahminler üretiliyor...")
    
    results = []
    
    for i in tqdm(range(0, len(test_ids), BATCH_SIZE)):
        batch_ids = test_ids[i : i + BATCH_SIZE]
        batch_seqs = [test_seqs[pid] for pid in batch_ids]
        
        encoded_batch = [processor.encode_sequence(seq) for seq in batch_seqs]
        input_tensor = torch.tensor(np.array(encoded_batch), dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        for j, pid in enumerate(batch_ids):
            protein_probs = probs[j]
            valid_indices = np.where(protein_probs > THRESHOLD)[0]
            
            for idx in valid_indices:
                term = int_to_term.get(idx)
                if term:
                    score = protein_probs[idx]
                    results.append(f"{pid}\t{term}\t{score:.3f}")

    # --- 5. DOSYAYI YAZ ---
    print(f"\n💾 Dosyaya yazılıyor ({len(results)} satır)...")
    with open(submission_path, "w") as f:
        f.write("\n".join(results))
        
    print(f"✅ BİTTİ! Dosya hazır: {submission_path}")

if __name__ == "__main__":
    make_submission()