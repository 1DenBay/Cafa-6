import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import os
import sys
from tqdm import tqdm

# --- AYARLAR ---
BATCH_SIZE = 8          # GPU patlamasın diye düşük (BERT çok RAM yer)
ACCUMULATION_STEPS = 4  # Sanal olarak Batch Size'ı 32 gibi çalıştırır
NUM_LABELS = 1500
LEARNING_RATE = 2e-5    # İnce ayar (Fine-tuning) için yavaş hız
EPOCHS = 3              # BERT çok zekidir, 3 turda bile öğrenir (Vakit kazanmak için)
MAX_LEN = 512           # Protein okuma limiti

# Proje Kök Dizini
if os.path.exists("/kaggle/working/Cafa-6"):
    project_root = "/kaggle/working/Cafa-6"
else:
    project_root = os.getcwd()

sys.path.append(project_root)
# DataProcessor'ı çağırıyoruz (Eski dostumuz)
from src.data_processor import CafaProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET SINIFI (BERT FORMATI) ---
class ProtBertDataset(Dataset):
    def __init__(self, sequences, labels_df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids = list(sequences.keys())
        self.seqs = sequences
        
        # Etiketleri hızlı bulmak için sözlük yapıyoruz
        # Groupby biraz yavaş olabilir ama en güvenli yoldur
        print("    ⚙️ Etiketler haritalanıyor...")
        self.labels_map = labels_df.groupby("EntryID")["term_idx"].apply(list).to_dict()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        seq = self.seqs[pid]
        
        # ProtBERT proteinleri "M A L ..." şeklinde boşluklu sever
        seq_spaced = " ".join(list(seq))
        
        # Harfleri sayıya çevir (Tokenize)
        encoding = self.tokenizer.encode_plus(
            seq_spaced,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Hedef Etiket (One-Hot)
        label_vec = torch.zeros(NUM_LABELS)
        if pid in self.labels_map:
            for i in self.labels_map[pid]:
                if i < NUM_LABELS:
                    label_vec[i] = 1.0
            
        return input_ids, attention_mask, label_vec

# --- MODEL SINIFI ---
class CafaProtBert(nn.Module):
    def __init__(self, num_labels=1500):
        super(CafaProtBert, self).__init__()
        # ProtBERT'i indiriyoruz (Yaklaşık 1.6 GB)
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        
        # DONDURMA (Freezing): Bert'in beynini donduruyoruz, sadece son katmanı eğitiyoruz.
        # Bu sayede eğitim 10 kat hızlanır ve GPU yetmezliği yaşamazsın.
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.3)
        # Sınıflandırıcı Katman
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), # ProtBERT çıktısı 1024'tür
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # Cümle özeti (CLS token)
        x = self.dropout(pooled_output)
        return self.classifier(x)

# --- ANA EĞİTİM FONKSİYONU ---
def train_bert():
    print(f"🚀 Cihaz: {device}")
    print("🚀 SİSTEM: PROT-BERT MODU")
    
    # 1. Dosyaları Bul (Kaggle Klasörlerinde)
    kaggle_input = "/kaggle/input"
    train_fasta = None
    train_terms = None
    
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            for file in files:
                if "train_sequences" in file and file.endswith(".fasta"):
                    train_fasta = os.path.join(root, file)
                elif "train_terms" in file and file.endswith(".tsv"):
                    train_terms = os.path.join(root, file)
    
    # Veri İşleyiciyi Hazırla
    processor = CafaProcessor(project_root=project_root, num_labels=NUM_LABELS)
    
    if train_fasta and train_terms:
        processor.fasta_path = train_fasta
        processor.terms_path = train_terms
        print(f"✅ Kaggle verisi tespit edildi.")
    else:
        print("⚠️ Veri bulunamadı! Lütfen Kaggle Input'u kontrol et.")
        # Devam edersek hata alırız, o yüzden burada durmuyoruz, data_processor hata verecek zaten.

    print("📊 Veriler yükleniyor...")
    df_terms = processor.load_labels()
    train_seqs = processor.load_fasta()
    
    print("📥 Tokenizer indiriliyor (İnternet açık olmalı)...")
    try:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    except:
        print("❌ HATA: Model indirilemedi! Kaggle'da İnternet'i açtın mı?")
        return
    
    # Dataset ve Loader
    full_dataset = ProtBertDataset(train_seqs, df_terms, tokenizer, MAX_LEN)
    
    # %90 Train, %10 Val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("🧠 ProtBERT Modeli Hafızaya Alınıyor...")
    model = CafaProtBert(num_labels=NUM_LABELS).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() # Hızlandırıcı
    
    best_f1 = 0.0
    os.makedirs(f"{project_root}/models", exist_ok=True)
    # DİKKAT: Dosya ismini farklı veriyoruz ki eski model silinmesin
    save_path = f"{project_root}/models/best_protbert_model.pth"
    
    print(f"\n🔥 EĞİTİM BAŞLIYOR ({EPOCHS} Epoch)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # İlerleme çubuğu
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (ids, mask, labels) in enumerate(loop):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(ids, mask)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * ACCUMULATION_STEPS
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
            
        # Validation (Her epoch sonu kontrol)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(ids, mask)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.25).float() # Threshold
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        # Basit F1
        tp = (all_preds * all_targets).sum()
        fp = (all_preds * (1 - all_targets)).sum()
        fn = ((1 - all_preds) * all_targets).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        print(f"Epoch {epoch+1} Bitti -> Ort. Loss: {total_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"    💾 KAYDEDİLDİ: {save_path} (Skor: {best_f1:.4f})")

if __name__ == "__main__":
    train_bert()