# 🧬 BioTahmin AI: Protein Fonksiyon Keşfi

 **BioTahmin AI** , biyolojik protein dizilimlerini (FASTA formatında) analiz ederek, bu proteinlerin hücresel işlevlerini (Gene Ontology Terms) tahmin eden uçtan uca bir yapay zeka uygulamasıdır.

CAFA-6 (Critical Assessment of Functional Annotation) veri seti kullanılarak eğitilen bu sistem, **Hibrit Derin Öğrenme (CNN + LSTM)** mimarisi sayesinde yüksek doğruluk ve performans sunar.

---

## 🚀 Özellikler

* **Kullanıcı Dostu Arayüz:** Streamlit tabanlı modern web arayüzü ile kod bilgisi gerektirmez.
* **Hibrit Model Mimarisi:** Görsel desenleri yakalamak için **ResNet (CNN)** ve sıralı dizilim ilişkilerini çözmek için **Bi-LSTM** kullanır.
* **Hızlı ve Hafif:** Büyük Dil Modellerine (LLM) kıyasla çok daha hızlıdır ve CPU üzerinde bile rahatlıkla çalışır.
* **Etkileşimli Analiz:** Kullanıcı tarafından ayarlanabilir **Güven Eşiği (Confidence Threshold)** ile analiz hassasiyetini yönetebilirsiniz.
* **Raporlama:** Sonuçları anında görüntüler ve detaylı analiz raporunu **Excel (CSV)** formatında indirmenizi sağlar.
* **Veri Gizliliği:** Tüm analiz yerel makinede (Localhost) yapılır, veriler buluta gönderilmez.

---

## 🧠 Model Mimarisi

Bu proje, biyolojik sekans verilerini işlemek için özel olarak tasarlanmış özgün bir mimari kullanır:

1. **Embedding Layer:** Amino asitleri 22 boyutlu vektör uzayına taşır.
2. **ResNet (1D CNN) Blokları:** Proteindeki yerel motifleri ve desenleri (Kernel Size: 9) yakalar. Dip katmanlara bozulmadan ulaşmasını sağlar.
3. **Bi-Directional LSTM:** Proteinin başından sonuna ve sonundan başına olan bağlamı öğrenir. Hafıza özelliği kazandırır.
4. **Global Max Pooling:** En belirgin özellikleri seçer. Önemine göre değerlendirir.
5. **Classifier:** 1500 farklı GO Terimi için olasılık üretir.

---

## 🛠️ Kurulum

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimler

Python 3.9 veya üzeri kurulu olmalıdır.

**Bash**

```
git clone https://github.com/1DenBay/Cafa-6.git
cd Cafa-6
```

### 2. Kütüphanelerin Yüklenmesi

Gerekli bağımlılıkları yükleyin:

**Bash**

```
pip install -r requirements.txt
```

*(Eğer requirements.txt dosyanız yoksa manuel olarak: `pip install streamlit torch pandas biopython`)*

### 3. Model Dosyası

Eğitilmiş model dosyasını (`best_cafa_model.pth`) projenin `models/` klasörüne yerleştirin.

---

## ▶️ Kullanım

Uygulamayı başlatmak için sadece terminalde şu komutu çalıştırın:

**Bash**

```
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılan sayfada:

1. Sol panelden **Güven Eşiğini** ayarlayın.
2. FASTA formatındaki dosyanızı sürükleyip bırakın.
3. **"Analizi Başlat"** butonuna basın.
4. Sonuçları inceleyin ve raporu indirin.

---

## 📂 Proje Yapısı

**Plaintext**

```
CAFA-6/
│
├── models/
│   └── best_cafa_model.pth        # Eğitilmiş PyTorch Modeli
│   └── cafa_model_protbert.pth    # (Private) İleri Seviye PyTorch Modeli
│
├── notebooks/
│   └── veri_kesif.ipynb           # Model, Sistem Testleri
│
├── src/
│   ├── train.py               		# Eğitim kodları
│   └── model.py              		# Model mimarisi
│   └── predict.py             		# Model testleri
│   └── submission.py           	# Model Geçerlilik Testi için .tvs Çıktısı
│   └── train_bert.py           	# Kullanılmayan 2.Model Mimarisi
│   └── data_downloader.py              # Verileri İstenen Formatta İndirir
│   └── data_processor.py               # Verileri İşlenebilecek Vektörlere Dönüştürür
│
├── data/                      # Örnek veriler
│
├── app.py                     # Streamlit Uygulaması (Main)
├── README.md                  # Dokümantasyon
└── requirements.txt           # Bağımlılıklar
```

---

## 📊 Performans

Model, CAFA-6 yarışması validasyon setinde aşağıdaki başarımları göstermiştir:

* **Validation F1-Score:** ~0.22 (Top Tier Performance)
* **Inference Hızı:** ~0.05 saniye/protein (CPU)

---

## 👤 İletişim

Bu proje **Deniz BAYAT** tarafından geliştirilmiştir.  *-Teşekkürler, Saygılar*

* LinkedIn: linkedin.com/in/denizbayat1/
* GitHub: github.com/1DenBay
* Medium: medium.com/@denizbyat
* Email: denizbyat@gmail.com
