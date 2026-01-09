# 1D CNN modeli
import torch
import torch.nn as nn

# --- [YENİ] 1. RESNET BLOĞU (Akıllı Tuğla) ---
# Bu sınıfı ana modelin içinde defalarca kullanacağız.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super(ResidualBlock, self).__init__()
        
        # A. ANA YOL (İşlem Hattı)
        # Conv -> BatchNorm -> ReLU -> Dropout -> Conv -> BatchNorm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels) # [YENİ] Veriyi normalize eder (Hizalar)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # B. KESTİRME YOL (Shortcut / Identity)
        # Eğer giriş (örn: 128) ve çıkış (örn: 256) sayıları farklıysa,
        # ham veriyi de toplamak için boyutunu eşitlememiz gerekir (1x1 Conv ile).
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1), # Boyut eşitleyici
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 1. Ham veriyi sakla (Kestirme yol için)
        identity = x 
        
        # 2. Ana yoldan geçir
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 3. [KRİTİK NOKTA] Ana Yol + Kestirme Yol
        # İşlenmiş veriye, orijinal veriyi (gerekirse boyutu eşitlenmiş) ekliyoruz.
        out += self.shortcut(identity)
        
        # 4. Son aktivasyon
        out = self.relu(out)
        
        return out
class CafaCNN(nn.Module):
    def __init__(self, num_labels=1500, vocab_size=22, embed_dim=128):
        super(CafaCNN, self).__init__()
        
        # 1. EMBEDDING KATMANI (Harfleri 'Kimyasal Manzaraya' Çevirme)
        # Girdi: [3, 15, ...] (Sayılar)
        # Çıktı: Her sayı için 128 özellikli bir vektör.
        # padding_idx=0 -> "0" ile doldurduğumuz kısımları 'yok' sayar.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. ResNet Katmanları (Eski Conv1/Conv2 yerine)
        # 128 özellik girer -> 256 özellik çıkar (Derinleşiyor)
        self.layer1 = ResidualBlock(in_channels=embed_dim, out_channels=256)
        
        # 256 özellik girer -> 512 özellik çıkar (Daha da derinleşiyor)
        self.layer2 = ResidualBlock(in_channels=256, out_channels=512)
        
        # --- 3. LSTM KATMANI (Hafıza) ---
        # ResNet'den çıkan 512 özelliği alır, zaman içindeki ilişkisine bakar.
        # hidden_size=128: Hafıza kapasitesi.
        # bidirectional=True: Çift yönlü okuma (Hem ileri hem geri).
        self.lstm = nn.LSTM(input_size=512, 
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
        # 4. SINIFLANDIRMA (Karar Verme)
        # Çıkarılan özete bakıp 1500 farklı etiket ile tahminlerini (olasılıklarını) üretir.
        # her etiket için bir skor üretilmiştir. Skor yükseliğine göre uyum da o kadar yüksektir.
        self.classifier = nn.Linear(128 * 2, num_labels)

    def forward(self, x):
        # x'in şekli: [Batch_Size, 1024] (Örn: 32 protein)
        
        # Adım 1: Sayıları vektöre çevir
        x = self.embedding(x)  
        # Şekil: [Batch, 1024, 128]
        x = x.permute(0, 2, 1)          # -> [Batch, 128, 1024] (CNN formatı)

        # --- [YENİ] ResNet Blokları ---
        # Artık "self.conv1(x)" demiyoruz, bloğa gönderiyoruz, o içeride hallediyor.
        x = self.layer1(x)              # -> [Batch, 256, 1024]
        x = self.layer2(x)              # -> [Batch, 512, 1024]
        
        # LSTM BÖLÜMÜ
        # 1. Boyut Çevirme: LSTM, zaman boyutunu (1024) ortada ister.
        # (Batch, Özellik, Zaman) -> (Batch, Zaman, Özellik)
        x = x.permute(0, 2, 1)          # -> [Batch, 1024, 256]
        
        # 2. Hafızaya Alma
        # output: Tüm zaman adımlarındaki hafıza durumu
        output, (hn, cn) = self.lstm(x) # -> [Batch, 1024, 256] (128 ileri + 128 geri)
        
        # 3. Özetleme (Max Pooling over Time)
        # 1024 adımın tamamına bak, en güçlü sinyali al.
        # (dim=1 yani zaman ekseninde maksimumu alıyoruz)
        x, _ = torch.max(output, dim=1) # -> [Batch, 256]
        
        # Son Karar
        logits = self.classifier(x) 
        # Şekil: [Batch, num_labels] -> Her etiket için bir puan
        
        return logits