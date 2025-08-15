# LUMINEUX-FarmaBot

## Proje Açıklaması
FarmaBot, ilaç prospektüs bilgilerini kullanarak kullanıcıların ilaç ile ilgili sorularını yanıtlayan hibrit yapay zeka sistemidir. BERT tabanlı derin öğrenme modeli ve kural bazlı yaklaşımları birleştirerek doğru ve güvenilir bilgi sağlar.

## Özellikler

### Ana Fonksiyonlar
- **Hibrit AI Sistemi**: BERT + Context-based yaklaşım
- **Çoklu Intent Tanıma**: 9 farklı soru kategorisi
- **Gerçek Zamanlı Soru-Cevap**: Web tabanlı kullanıcı arayüzü
- **Feedback Sistemi**: Kullanıcı geri bildirim toplama
- **Güven Skoru**: Her cevap için güvenilirlik puanı

### Sağladığı Bilgiler
- İlaç tanımı ve kullanım alanları
- Doz bilgileri ve kullanım şekli
- Yan etkiler ve uyarılar
- Hamilelik ve emzirme döneminde kullanım
- İlaç etkileşimleri
- Saklama koşulları
- Dikkat edilmesi gereken durumlar

## Hızlı Başlangıç

### Gereksinimler
```bash
Python 3.8+
Flask
torch
transformers
numpy
```

### Kurulum
1. **Depoyu klonlayın**
```bash
[git clone https://github.com/username/farmabot.git](https://github.com/iremNur-Demiray/LUMINEUX-FarmaBot.git)
cd FarmaBot
```

2. **Sanal ortam oluşturun**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

3. **Gerekli paketleri yükleyin**
```bash
pip install flask torch transformers numpy
```

4. **Veri dosyalarını yerleştirin**
```
FarmaBot/
├──  app.py
├──  Veri Seti/
│   └──  Dataset_.json
├──  Model/
│   └──  bert_matching_model/
│       └──  best_model/
└──  Logo/
    └──  FarmaBotLogo.png
```

### Çalıştırma
```bash
python app.py
```

Tarayıcınızda `http://localhost:5000` adresine gidin.

##  Proje Yapısı

```
FarmaBot/
├──  app.py                    # Ana uygulama dosyası
├── README.md                 # Bu dosya
├──  Veri Seti/
│   └──  Dataset_.json         # İlaç veritabanı
├──  Model/
│   └──  bert_matching_model/
│       └──  best_model/       # Eğitilmiş BERT modeli
├──  Logo/
│   └──  FarmaBotLogo.png      # Logo dosyası
└──  GeriBildirim.json         # Kullanıcı geri bildirimleri
```

##  Konfigürasyon

### Model Ayarları
`app.py` dosyasının en altında bulunan ayarları düzenleyin:

```python
# Model parametreleri
data_path = r"C:\path\to\your\Dataset_.json"          # İlaç veri dosyası - ZORUNLU
bert_model_path = r"C:\path\to\your\bert_model"       # BERT model yolu - OPSİYONEL
```

### Veri Dosyası Formatı
İlaç verisi JSON formatında olmalıdır:

```json
{
  "ilaç_adı": {
    "nedir_ve_ne_icin_kullanilir": "İlaç açıklaması...",
    "uygun_kullanim_doz_siklik": "Doz bilgileri...",
    "olasi_yan_etkiler": "Yan etkiler...",
    "hamilelikte_kullanim": "Hamilelik bilgileri...",
    "emzirirken_kullanim": "Emzirme bilgileri...",
    "etkilesimler": "Etkileşim bilgileri...",
    "saklama_ve_muhafaza": "Saklama koşulları...",
    "kullanilmamasi_gereken_durumlar": "Kontrendikasyonlar...",
    "dikkatli_kullanim_gerektiren_durumlar": "Uyarılar..."
  }
}
```

##  AI Model Detayları

### Hibrit Yaklaşım
1. **BERT Intent Classification**: Sorunun hangi kategoride olduğunu belirler
2. **Context-based Matching**: İlaç adı ve ilgili bölümü bulur
3. **Smart Answer Generation**: Kontekst'ten uygun cevap üretir

### Intent Kategorileri
- `nedir_ve_ne_icin_kullanilir`
- `uygun_kullanim_doz_siklik`
- `olasi_yan_etkiler`
- `hamilelikte_kullanim`
- `emzirirken_kullanim`
- `etkilesimler`
- `saklama_ve_muhafaza`
- `kullanilmamasi_gereken_durumlar`
- `dikkatli_kullanim_gerektiren_durumlar`

### Güven Skoru Hesaplama
- BERT confidence (>= 0.8 için bonus)
- Cevap uzunluğu kontrolü
- Context kaynak türü değerlendirmesi

##  API Endpoints

### POST /api/ask
Soru sorma endpoint'i
```json
{
  "question": "Parol ile alkol alınır mı?",
  "user_id": "anonymous"
}
```

**Yanıt:**
```json
{
  "id": "uuid",
  "question": "Parol ile alkol alınır mı?",
  "answer": "Parol ile alkol...",
  "confidence": 0.85,
  "detected_drug": "parol",
  "model_version": "hybrid-bert-context-v1.0",
  "method": "BERT+Context",
  "bert_intent": "etkilesimler",
  "bert_confidence": 0.92
}
```

### POST /api/feedback
Geri bildirim gönderme
```json
{
  "question": "Soru metni",
  "model_answer": "Model cevabı",
  "feedback_type": "like/dislike",
  "reason": "helpful/wrong_answer/incomplete",
  "comment": "Ek yorum"
}
```

### GET /api/feedback/list
Tüm geri bildirimleri listele (admin)

## Web Arayüzü

### Özellikler
- **Responsive Design**: Mobil ve masaüstü uyumlu
- **Real-time Chat**: Anlık soru-cevap
- **Animated Loading**: Kapsül açılma animasyonu
- **Feedback System**: Beğeni/beğenmeme butonları
- **Character Counter**: 500 karakter sınırı
- **Fixed Sidebar**: Sol panelde yardım bilgileri

### Teknolojiler
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: TailwindCSS
- **Backend**: Flask (Python)

##  Gelişmiş Konfigürasyon

### BERT Model Gereksinimleri
- Torch 1.9+
- Transformers 4.0+
- Model dosyaları: `config.json`, `pytorch_model.bin`, `tokenizer.json`

### Performans Optimizasyonu
- GPU kullanımı otomatik tespit edilir
- Model cache mekanizması
- Async processing için gelecek güncellemeler

### Logging
```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```



##  Test Etme

### Örnek Sorular
```
✅ "Parol nedir ve ne için kullanılır?"
✅ "Aspirin ile alkol alınır mı?"
✅ "Hamilelikte antibiyotik kullanımı"
✅ "İbuprofen yan etkileri nelerdir?"
✅ "Omeprazol nasıl saklanmalıdır?"
```

### Debug Modu
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

##  Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişiklikleri commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

### Geliştirme Rehberi
- PEP 8 kod standardını takip edin
- Type hints kullanın
- Docstring'leri güncel tutun
- Unit testler ekleyin






## İletişim  

### Teknik Destek
- **Email**: iremnur.demiray86@erzurum.edu.tr
- İREM NUR DEMİRAY

- **Email**: dilara.adiguzel66@erzurum.edu.tr
- DİLARA ADIGÜZEL


*Son güncelleme: 2025*
