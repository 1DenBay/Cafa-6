# 1D CNN modeli
import torch
import torch.nn as nn

class CafaCNN(nn.Module):
    def __init__(self, num_labels=1500, vocab_size=22, embed_dim=128):
        super(CafaCNN, self).__init__()
        
        # 1. EMBEDDING KATMANI (Harfleri 'Kimyasal Manzaraya' Çevirme)
        # Girdi: [3, 15, ...] (Sayılar)
        # Çıktı: Her sayı için 128 özellikli bir vektör.
        # padding_idx=0 -> "0" ile doldurduğumuz kısımları 'yok' sayar.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. KONVOLÜSYON KATMANI (Büyüteç / Desen Arayıcı)
        # BLOK 1    
        # in_channels=128 : Embedding'den gelen özellik sayısı.
        # out_channels=256 : Modelin arayacağı farklı desen sayısı (256 farklı filtre). Yani aynı anda 256 farklı özellik araması yapar.
        # kernel_size=9   : Büyüteç genişliği. Yan yana 9 amino asidi aynı anda inceler.
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # %30 Unutkanlık (Ezber bozucu)
        
        # BLOK 2
        # İlk katmanın bulduğu 256 özelliği alır -> 512 daha derin özelliğe çevirir
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, padding=3)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3) # Yine %30 Unutkanlık
        
        # 3. POOLING (Özetleme)
        # AdaptiveMaxPool1d(1) -> Tüm dizi boyunca bulunan 'en güçlü' sinyali alır.
        # 1024 uzunluğundaki diziyi tek ve en güçlü bir vektöre indirger.
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 4. SINIFLANDIRMA (Karar Verme)
        # Çıkarılan özete bakıp 1500 farklı etiket ile tahminlerini (olasılıklarını) üretir.
        # her etiket için bir skor üretilmiştir. Skor yükseliğine göre uyum da o kadar yüksektir.
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, x):
        # x'in şekli: [Batch_Size, 1024] (Örn: 32 protein)
        
        # Adım 1: Sayıları vektöre çevir
        x = self.embedding(x)  
        # Şekil: [Batch, 1024, 128]
        
        # Adım 2: Boyutları Düzenle (PyTorch CNN kuralı) -> konvülasyon adımı için gereklidir
        # PyTorch'un CNN katmanı, kanalları (128 özelliği) ortada ister.
        # (Batch, Uzunluk, Özellik) -> (Batch, Özellik, Uzunluk) yapıyoruz.
        x = x.permute(0, 2, 1) 
        # Şekil: [Batch, 128, 1024]
        
        # BLOK 1 İŞLEMLERİ
        x = self.conv1(x)    # Desen ara
        x = self.relu1(x)     # Gürültüyü sil
        x = self.dropout1(x)   # Unutkanlık (Ezber bozma)
        # Şekil: [Batch, 256, 1]

        # BLOK 2 İŞLEMLERİ
        x = self.conv2(x)               # -> [Batch, 512, 1024]
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Adım 3: Düzleştirme ve özetleme (Gereksiz boyutu at)
        x = self.pool(x)                # -> [Batch, 512, 1]
        x = x.squeeze(-1) 
        # Şekil: [Batch, 256]
        
        # Adım 4: Son Karar
        logits = self.classifier(x) 
        # Şekil: [Batch, num_labels] -> Her etiket için bir puan
        
        return logits