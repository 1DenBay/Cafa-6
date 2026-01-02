import os
import zipfile
import subprocess
# Terminal komutu yerine doğrudan Python API kullanıyoruz
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data_native():
    # Kaggle.json dosyasını şu an olduğumuz yerde (Cafa-6 klasöründe) ara
    current_dir = os.getcwd()
    os.environ['KAGGLE_CONFIG_DIR'] = current_dir
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Anahtar Yeri: {os.path.join(current_dir, 'kaggle.json')}")
    print("Kaggle API'ye bağlanılıyor...")

    try:
        # İNDİRME
        api = KaggleApi()
        api.authenticate() # Kimlik doğrula
        
        print(f"İndirme başladı: {output_dir} (Bu biraz sürebilir)...")
        
        # Yarışma dosyalarını indir
        api.competition_download_files('cafa-6-protein-function-prediction', path=output_dir, quiet=False)
        
        print("İndirme Tamamlandı! Zip açılıyor...")

        # ZİP AÇMA
        zip_path = os.path.join(output_dir, "cafa-6-protein-function-prediction.zip")
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path) # Ana zipi sil
            
            # İçerdeki diğer zipleri de aç (Mac uyumlu)
            print("Alt dosyalar açılıyor...")
            subprocess.run(f"unzip -o -q '{output_dir}/*.zip' -d {output_dir} 2>/dev/null", shell=True)
            
            # .gz dosyalarını aç
            subprocess.run(f"find {output_dir} -name '*.gz' -exec gunzip -f {{}} \;", shell=True)
            
            print("TERTEMİZ BİTTİ! Dosyalar hazır.")
        else:
            print("⚠️ Dosya indi görünüyor ama zip dosyası bulunamadı.")

    except Exception as e:
        print(f"HATA: {e}")
        print("Lütfen 'kaggle.json' dosyasının 'Cafa-6' klasörünün içinde olduğundan emin ol.")

if __name__ == "__main__":
    download_data_native()