# Teknofest Sağlıkta Yapay Zeka Yarışması - 2. Aşama Model Geliştirme

Bu repo, **Teknofest Sağlıkta Yapay Zeka Yarışması**nın 2. aşamasında kullanılan modellerin eğitim kodlarını, ağırlıklarını ve sonuçlarını içermektedir. Çalışmada, farklı medikal görüntüleme teknikleriyle eğitilen modeller ve bu modellerin topluluk (ensemble) yöntemleriyle birleştirilmiş sonuçları sunulmaktadır.

## 📁 Klasör Yapısı

├── model_training_code/  
├── model_weights/  
└── results/

### 1. `model_training_code/`

Bu klasör, dört farklı görüntüleme modalitesine göre özelleştirilmiş eğitim kodlarını içermektedir:

- `modality` ct, dwi, t2wi, adc  için özelleştirilmiş **ConvNeXt-Base** modelinin eğitim kodu.
- `early_ensemble`: Farklı modalitelerin Global Average Pooling katmanından alınan öznitelikleri birleştirerek **vakabazlı MLP katmanı** ile yapılan early ensemble ve kombinasyonları için kodlar.
- `late_ensemble`: Farklı modalitelerin çıktılarını birleştirerek **ortalama** ile yapılan late ensemble ve kombinasyonları için kodlar.
  
### 2. `model_weights/`

Bu klasörde eğitilmiş modellerin ağırlıkları yer almaktadır:

- Dört modaliteye ait bireysel model ağırlıkları.
- `early_ensemble` MLP için ağırlık dosyaları.

### 3. `results/`

Bu klasör, eğitim ve test verileriyle elde edilen sonuçların görselleştirmelerini içerir:

- `modalite/`: Her bir modalite için ROC eğrileri, confusion matrix ve t-SNE grafikleri.
- `early_ensemble/`: Early ensemble sonuçlarına ait ROC, confusion matrix ve t-SNE görselleri.
- `late_ensemble/`: Late ensemble sonuçlarına ait ROC, confusion matrix ve t-SNE görselleri.

## 🔧 Gereksinimler

Projeyi çalıştırmadan önce aşağıdaki paketlerin yüklü olması gerekmektedir. İlgili `requirements.txt` dosyasını kullanabilirsiniz:

```bash
pip install -r requirements.txt
