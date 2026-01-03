# PROTEİNLERİ (HARFLERİ) SAYILARA ÇEVİRME (MULTİ-HOT)
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class CafaProcessor:
    def __init__(self, project_root=None, num_labels=1500, max_len=1024):
        if project_root is None:
            self.project_root = os.getcwd()
        else:
            self.project_root = project_root
            
        self.num_labels = num_labels
        self.max_len = max_len

        # Amino Asit Sözlüğü
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i+1 for i, aa in enumerate(self.amino_acids)}

        # Dosya Yolları
        base_raw = os.path.join(self.project_root, "data", "raw")
        if os.path.exists(os.path.join(base_raw, "Train", "train_terms.tsv")):
            self.train_dir = os.path.join(base_raw, "Train")
        else:
            self.train_dir = base_raw
            
        self.terms_path = os.path.join(self.train_dir, "train_terms.tsv")
        self.fasta_path = os.path.join(self.train_dir, "train_sequences.fasta")
        
        # Etiket haritaları
        self.top_terms = []
        self.term_to_int = {}

    def load_labels(self):
        print(f"Etiket dosyası okunuyor: {self.terms_path}")
        if not os.path.exists(self.terms_path):
            raise FileNotFoundError(f"HATA: {self.terms_path} yok!")

        train_terms = pd.read_csv(self.terms_path, sep="\t")

        # En yaygın N etiketi seç
        print(f"En yaygın {self.num_labels} etiket seçiliyor...")
        counts = train_terms['term'].value_counts()
        self.top_terms = counts.head(self.num_labels).index.tolist()
        self.term_to_int = {term: i for i, term in enumerate(self.top_terms)}
        
        # Sadece seçilen etiketleri içeren satırları tut (Hızlandırma)
        train_terms = train_terms[train_terms['term'].isin(self.top_terms)]
        print("Etiket haritası hazır.")
        return train_terms

    def encode_sequence(self, sequence):
        encoded = [self.aa_to_int.get(aa, 21) for aa in sequence]
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded += [0] * (self.max_len - len(encoded))
        return np.array(encoded, dtype=np.int32)

    def load_fasta(self):
        """Fasta dosyasını okuyup {ProteinID: Sequence} sözlüğü yapar"""
        print(f"Fasta okunuyor: {self.fasta_path}")
        seqs = {}
        current_id = None
        current_seq = []
        
        with open(self.fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"): # baştaki ">" işareti yeni bir proteini gösterir bunu referans alacağız geçiştlerde
                    if current_id:
                        seqs[current_id] = "".join(current_seq)
                    # Başlık örneği: >sp|A0A0C5B5G6|MOTSC_HUMAN...
                    # '|' işaretine göre bölüyoruz -> ['>sp', 'A0A0C5B5G6', 'MOTSC_HUMAN...']
                    # 1. indeksi (ortadaki ID'yi) alıyoruz.
                    parts = line.split('|') #proteinlerde boşluklar yerine düz çizgi ayracı kullanıulmış ona göre ayarlama yapacağız
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        # Eğer format farklıysa eski yöntemi dene (Yedek plan)
                        current_id = line[1:].split()[0]
            if current_id:
                seqs[current_id] = "".join(current_seq)
        
        print(f"{len(seqs)} protein hafızaya alındı.")
        return seqs

# --- GARSON (DATASET) SINIFI --- Verilen amino asit dizisini (proteini) processor ile sayılara çevirir sonrasında onuda vektörlere çevirir
class CafaDataset(Dataset):
    def __init__(self, processor, protein_ids, sequences, terms_df):
        self.processor = processor
        self.protein_ids = protein_ids # Liste: ['P123', 'P456'...]
        self.sequences = sequences     # Sözlük: {'P123': 'MACL...'}
        
        # Hangi proteinin hangi etiketlere sahip olduğunu hızlı bulmak için grupluyoruz
        print("⚙️ Veriler eşleştiriliyor (Etiketleme)...")
        self.labels_map = terms_df.groupby('EntryID')['term'].apply(list).to_dict()

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        # Proteini Bul
        p_id = self.protein_ids[idx]
        seq_str = self.sequences[p_id]
        
        # Diziyi Sayıya Çevir - ilk tanımlanan fonksiyonla
        input_ids = self.processor.encode_sequence(seq_str)
        
        # Etiketleri Vektöre Çevir (0 ve 1'ler)
        label_vector = np.zeros(self.processor.num_labels, dtype=np.float32)
        
        if p_id in self.labels_map:
            for term in self.labels_map[p_id]:
                if term in self.processor.term_to_int:
                    idx_label = self.processor.term_to_int[term]
                    label_vector[idx_label] = 1.0
        
        # PyTorch Tensor'una çevirip tepsiye koy
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_vector, dtype=torch.float32)
        }

"""
sinir ağları harf yada metinlerden anlamadığından proteinini GO:0005515 gibi etiketlerini sayılara kodladık. "data_processor.py" dosyası bizim çevirmenimiz oldu.

Proteinler 20 farklı harfden oluştuğundan her harf için değişmez sayı atandı
Model değişmez uzunlukta boyut ve vektör ister. yani proteinler önceki grafikte gördüğümüz gibi çok değişken boyuttu o yüzden ortalama 500 civarı olduğundan 
ve limiti 1024 uygun gördüğümüzden 1024den kısaların arkasına "0" ile doldurduk uzunlar içinde fazlasını kesip attık (saten 1024 den uzun olanlar çok az)
40 binden fazla etiket var modeli çökerteceğinden en çok geçen 1500 etiketi seçtik şimdilik
"""