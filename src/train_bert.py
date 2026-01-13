import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import os
import sys
import pandas as pd # Pandas ekledik
from tqdm import tqdm

# --- AYARLAR ---
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
NUM_LABELS = 1500
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LEN = 512

# Proje Kök Dizini
if os.path.exists("/kaggle/working/Cafa-6"):
    project_root = "/kaggle/working/Cafa-6"
else:
    project_root = os.getcwd()

sys.path.append(project_root)
from src.data_processor import CafaProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET ---
class ProtBertDataset(Dataset):
    def __init__(self, sequences, labels_df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids = list(sequences.keys())
        self.seqs = sequences
        
        # FIX: Groupby yapmadan önce sütun kontrolü
        if "term_idx" not in labels_df.columns:
            raise ValueError("HATA: Dataframe içinde 'term_idx' sütunu yok! Mapping yapılmamış.")
            
        print("    ⚙️ Etiketler haritalanıyor...")
        self.labels_map = labels_df.groupby("EntryID")["term_idx"].apply(list).to_dict()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        seq = self.seqs[pid]
        seq_spaced = " ".join(list(seq))
        
        encoding = self.tokenizer.encode_plus(
            seq_spaced, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        label_vec = torch.zeros(NUM_LABELS)
        if pid in self.labels_map:
            for i in self.labels_map[pid]:
                if i < NUM_LABELS: label_vec[i] = 1.0
            
        return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), label_vec

# --- MODEL ---
class CafaProtBert(nn.Module):
    def __init__(self, num_labels=1500):
        super(CafaProtBert, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        for param in self.bert.parameters(): param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.pooler_output))

# --- EĞİTİM ---
def train_bert():
    print(f"🚀 Cihaz: {device} (ProtBERT Modu)")
    
    # Kaggle Veri Yolları
    train_fasta, train_terms = None, None
    # Veri yollarını data/raw altından bulmaya çalış (Kaggle scripti oraya linkledi)
    if os.path.exists("data/raw/train_sequences.fasta"):
        train_fasta = "data/raw/train_sequences.fasta"
        train_terms = "data/raw/train_terms.tsv"
    
    processor = CafaProcessor(project_root, NUM_LABELS)
    if train_fasta: 
        processor.fasta_path = train_fasta
        processor.terms_path = train_terms
    
    print("📊 Veriler yükleniyor...")
    # Label yükleme ve FIX İŞLEMİ
    df_terms = processor.load_labels()
    
    # --- FIX: SÜTUN KONTROLÜ VE ZORLA OLUŞTURMA ---
    if "term_idx" not in df_terms.columns:
        print("⚠️ UYARI: 'term_idx' sütunu eksik, manuel oluşturuluyor...")
        # En çok geçen terimleri bul
        top_terms = df_terms["term"].value_counts().index[:NUM_LABELS]
        # Sözlük oluştur
        term_to_int = {term: i for i, term in enumerate(top_terms)}
        # Sadece top terms'i filtrele
        df_terms = df_terms[df_terms["term"].isin(top_terms)].copy()
        # Map işlemini yap
        df_terms["term_idx"] = df_terms["term"].map(term_to_int)
        print(f"✅ 'term_idx' sütunu oluşturuldu. Satır sayısı: {len(df_terms)}")
    # ---------------------------------------------

    train_seqs = processor.load_fasta()
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    
    dataset = ProtBertDataset(train_seqs, df_terms, tokenizer, MAX_LEN)
    train_size = int(0.9 * len(dataset))
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = CafaProtBert(NUM_LABELS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_f1 = 0.0
    os.makedirs(f"{project_root}/models", exist_ok=True)
    save_path = f"{project_root}/models/best_protbert_model.pth"
    
    print(f"🔥 EĞİTİM BAŞLIYOR ({EPOCHS} Epoch)...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (ids, mask, labels) in enumerate(loop):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                loss = criterion(model(ids, mask), labels) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(ids, mask)
                all_preds.append((torch.sigmoid(outputs) > 0.25).float().cpu())
                all_targets.append(labels.cpu())
        
        all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
        tp = (all_preds * all_targets).sum()
        f1 = 2 * tp / (2 * tp + (all_preds * (1 - all_targets)).sum() + ((1 - all_preds) * all_targets).sum() + 1e-8)
        print(f"Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"💾 Kaydedildi: {save_path}")

if __name__ == "__main__":
    train_bert()