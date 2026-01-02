# PROTEİNLERİ (HARFLERİNİ) SAYILARA ÇEVİRMEK (MULTİ-HOT)
import pandas as pd
import numpy as np
import os
from collections import Counter

class CafaProcessor:
    def __init__(self, project_root=None, num_labels=1500, max_len=1024):
        # Eğer root verilmezse, otomatik bulmaya çalış
        if project_root is None:
            self.project_root = os.getcwd()
        else:
            self.project_root = project_root
            
        self.num_labels = num_labels # Hedeflenecek en yaygın etiket sayısı
        self.max_len = max_len       # Protein dizisi uzunluk limiti

        # Amino Asit Sözlüğü (Harf -> Sayı)
        # 0: Padding (Boşluk), 21-25: Nadir/Bilinmeyenler
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i+1 for i, aa in enumerate(self.amino_acids)}

        # Hedef Etiketler (Daha sonra doldurulacak)
        self.top_terms = []
        self.term_to_int = {}
        
        # DOSYA YOLU AYARLAMA (Otomatik Algılama)
        # data/raw veya data/raw/Train kombinasyonlarını kontrol et
        base_raw = os.path.join(self.project_root, "data", "raw")
        
        if os.path.exists(os.path.join(base_raw, "Train", "train_terms.tsv")):
            self.terms_path = os.path.join(base_raw, "Train", "train_terms.tsv")
        else:
            self.terms_path = os.path.join(base_raw, "train_terms.tsv")

    def load_labels(self):
        """
        En yaygın 'num_labels' kadar etiketi seçer ve haritayı oluşturur.
        """
        print(f"Etiket dosyası okunuyor: {self.terms_path}")
        
        if not os.path.exists(self.terms_path):
            raise FileNotFoundError(f"HATA: Dosya bulunamadı! {self.terms_path}")

        train_terms = pd.read_csv(self.terms_path, sep="\t")

        # Sadece en çok geçen N terimi al
        print(f"En yaygın {self.num_labels} etiket seçiliyor...")
        counts = train_terms['term'].value_counts()
        self.top_terms = counts.head(self.num_labels).index.tolist()

        # Etiket -> Index haritası (GO:0005515 -> 0 gibi)
        self.term_to_int = {term: i for i, term in enumerate(self.top_terms)}
        print("Etiket haritası hazır.")
        return train_terms

    def encode_sequence(self, sequence):
        """
        Protein dizisini sayı dizisine çevirir (Padding/Truncation dahil).
        Örn: "MACL..." -> [10, 1, 2, 8, 0, 0...]
        """
        # Harfleri sayıya çevir
        encoded = [self.aa_to_int.get(aa, 21) for aa in sequence] # 21: Bilinmeyen

        # Boyut Ayarlama (1024'e sabitleme)
        if len(encoded) > self.max_len:
            # Çok uzunsa kes
            encoded = encoded[:self.max_len]
        else:
            # Kısaysa sonuna 0 ekle (Padding)
            encoded += [0] * (self.max_len - len(encoded))

        return np.array(encoded, dtype=np.int32)

if __name__ == "__main__":
    # Kendi kendine test bloğu
    print("--- TEST MODU ---")
    try:
        # Script olarak çalıştırıldığında data klasörünü bulması için bir üst dizine bakması gerekebilir
        # Ama şimdilik olduğu yeri varsayalım
        processor = CafaProcessor()
        processor.load_labels()

        sample_seq = "MKAWLVLVGVL"
        encoded = processor.encode_sequence(sample_seq)
        print(f"\nTest Dizisi: {sample_seq}")
        print(f"Kodlanmış: {encoded[:15]}...") 
    except Exception as e:
        print(f"Test hatası: {e}")

"""
sinir ağları harf yada metinlerden anlamadığından proteinini GO:0005515 gibi etiketlerini sayılara kodladık. "data_processor.py" dosyası bizim çevirmenimiz oldu.

Proteinler 20 farklı harfden oluştuğundan her harf için değişmez sayı atandı
Model değişmez uzunlukta boyut ister. yani proteinler önceki grafikte gördüğümüz gibi çok değişken boyuttu o yüzden ortalama 500 civarı olduğundan ve limiti 1024 uygun gördüğümüzden 1024den kısaların arkasına "0" ile doldurduk uzunlar içinde fazlasını kesip attık (saten 1024 den uzun olanlar çok az)
40 binden fazla etiket var modeli çökerteceğinden en çok geçen 1500 etiketi seçtik şimdilik
"""