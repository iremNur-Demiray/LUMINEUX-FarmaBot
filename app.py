from flask import Flask, request, jsonify, render_template_string
import json
import os
import re
import uuid
from datetime import datetime
import time
import warnings
from typing import Dict, List, Optional, Tuple
import logging

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BERT için gerekli kütüphaneler
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    import numpy as np
    BERT_AVAILABLE = True
    print("✅ BERT kütüphaneleri yüklendi")
except ImportError as e:
    BERT_AVAILABLE = False
    print(f"⚠️ BERT kütüphaneleri yüklenemedi: {e}")

class HybridDrugAdvisorModel:
    """BERT + Context-based hibrit akıllı ilaç chatbot sistemi"""
    
    def __init__(self, data_path: str, bert_model_path: str = None):
        """
        Hibrit chatbot'u başlatır
        
        Args:
            data_path: İlaç verilerinin JSON dosya yolu
            bert_model_path: BERT modelinin bulunduğu klasör yolu
        """
        print("🚀 BERT + Context-Based Hibrit İlaç Chatbot başlatılıyor...")
        
        # İlaç verilerini yükle
        self.load_drug_data(data_path)
        
        # BERT modelini yükle
        self.bert_enabled = False
        self.bert_model = None
        self.bert_tokenizer = None
        self.intent_labels = []
        
        # BERT model yükleme - geliştirilmiş kontrol
        if BERT_AVAILABLE and bert_model_path is not None and bert_model_path.strip():
            try:
                # Dosya yolu kontrolü - normalize et
                bert_model_path = os.path.normpath(bert_model_path)
                print(f"🔍 BERT model yolu kontrol ediliyor: {bert_model_path}")
                
                if os.path.exists(bert_model_path) and os.path.isdir(bert_model_path):
                    self._load_bert_model_safe(bert_model_path)
                else:
                    print(f"⚠️ BERT model klasörü bulunamadı: {bert_model_path}")
                    print("💡 Sadece Context-based mod ile devam ediliyor")
            except Exception as e:
                print(f"⚠️ BERT model yüklenemedi: {e}")
                print("💡 Sadece Context-based mod ile devam ediliyor")
        else:
            print("💡 BERT model yolu belirtilmedi veya BERT kütüphaneleri mevcut değil")
            print("💡 Sadece Context-based mod aktif")
        
        # Model versiyonu
        self.model_version = "hybrid-bert-context-v1.0"
        self.dataset_version = "drug_leaflet_hybrid_v1"
        
        print("🎉 Hibrit Chatbot hazır!\n")
    
    def load_drug_data(self, data_path: str):
        """İlaç verilerini yükler - SADECE gerçek veri seti"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"İlaç veri dosyası bulunamadı: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # İlaç verilerini düzgün şekilde işle
            self.ilac_data = {}  # Dictionary olarak saklayacağız
            self.ilac_isimleri = []  # Arama için isim listesi
            
            print(f"📋 Ham veri anahtarları: {list(raw_data.keys())[:5]}...")  # Debug
            
            for ilac_adi, ilac_bilgisi in raw_data.items():
                # İlaç adını temizle ve normalize et
                clean_name = self._normalize_drug_name(ilac_adi)
                
                # İlaç bilgisini dictionary'ye ekle
                self.ilac_data[clean_name] = ilac_bilgisi
                
                # Arama için farklı varyasyonları ekle
                self.ilac_isimleri.extend(self._generate_name_variants(clean_name))
                
                # Debug: İlk birkaç ilacı göster
                if len(self.ilac_data) <= 3:
                    print(f"✅ İlaç yüklendi: '{ilac_adi}' -> '{clean_name}'")
            
            print(f"✅ {len(self.ilac_data)} ilaç verisi yüklendi")
            print(f"🔍 Toplam arama terimi: {len(set(self.ilac_isimleri))}")
            
        except Exception as e:
            raise Exception(f"Veri yüklenemedi: {e}")
    
    def _normalize_drug_name(self, name: str) -> str:
        """İlaç adını normalize eder"""
        if not name:
            return ""
        
        # Türkçe karakterleri düzelt ve küçük harfe çevir
        name = name.lower().strip()
        
        # Özel karakterleri temizle ama tire ve parantezleri koru
        name = re.sub(r'[^\w\s\-\(\)]+', '', name)
        
        return name
    
    def _generate_name_variants(self, drug_name: str) -> List[str]:
        """İlaç adının farklı varyasyonlarını üretir"""
        variants = set()
        
        # Orijinal adı ekle
        variants.add(drug_name)
        
        # Kelimelere böl
        words = drug_name.split()
        
        # İlk kelimeyi ekle
        if words:
            variants.add(words[0])
        
        # Parantez içindeki içerikleri de ekle
        parentheses_content = re.findall(r'\((.*?)\)', drug_name)
        for content in parentheses_content:
            variants.add(content.strip())
        
        # Tire ile ayrılmış kısımları ekle
        hyphen_parts = drug_name.split('-')
        for part in hyphen_parts:
            if len(part.strip()) > 2:
                variants.add(part.strip())
        
        # Sayıları kaldırılmış versiyonu ekle
        no_numbers = re.sub(r'\d+', '', drug_name).strip()
        if no_numbers and no_numbers != drug_name:
            variants.add(no_numbers)
        
        return list(variants)
    
    def _load_bert_model_safe(self, model_path: str):
        """BERT modelini güvenli şekilde yükler - vocab uyumsuzluğu çözümü ile"""
        print(f"🔄 BERT model güvenli yükleme başlıyor: {model_path}")
        
        try:
            # İlk olarak config'i yükle ve kontrol et
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                print("❌ config.json bulunamadı")
                return
            
            print("🔄 Model konfigürasyonu okunuyor...")
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            print(f"📊 Model config vocab size: {config.vocab_size}")
            
            # Tokenizer'ı yükle ve vocab boyutunu kontrol et
            print("🔄 Tokenizer yükleniyor...")
            try:
                # İlk önce modelin kendi tokenizer'ını dene
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")
            except Exception as tokenizer_error:
                print(f"⚠️ Model tokenizer hatası: {tokenizer_error}")
                print("🔄 Standart BERT tokenizer deneniyor...")
                
                # Standart BERT tokenizer'ı dene
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                print(f"📊 Standart tokenizer vocab size: {tokenizer.vocab_size}")
            
            # Vocab boyutu uyumsuzluğu kontrolü
            if hasattr(config, 'vocab_size') and config.vocab_size != tokenizer.vocab_size:
                print(f"⚠️ Vocab boyutu uyumsuzluğu tespit edildi!")
                print(f"   Model beklenen: {config.vocab_size}")
                print(f"   Tokenizer mevcut: {tokenizer.vocab_size}")
                
                # Config'i tokenizer'a uyacak şekilde güncelle
                print("🔄 Config vocab boyutu güncelleniyor...")
                config.vocab_size = tokenizer.vocab_size
                
            # Model'i yükle
            print("🔄 Model yükleniyor...")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    config=config,  # Güncellenmiş config'i kullan
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True  # Boyut uyumsuzluklarını görmezden gel
                )
                print("✅ Model başarıyla yüklendi (ignore_mismatched_sizes=True ile)")
                
            except Exception as model_error:
                print(f"⚠️ Model yükleme hatası: {model_error}")
                return
            
            # Model'i evaluation moduna al
            model.eval()
            
            # GPU varsa kullan
            if torch.cuda.is_available():
                model = model.cuda()
                print("🚀 Model GPU'ya taşındı")
            else:
                print("💻 Model CPU'da çalışıyor")
            
            # Intent etiketlerini yükle (varsa)
            label_file = os.path.join(model_path, 'intent_labels.json')
            intent_labels = []
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    intent_labels = json.load(f)
                print(f"✅ Intent etiketleri yüklendi: {len(intent_labels)} adet")
            else:
                print("⚠️ intent_labels.json bulunamadı, varsayılan etiketler kullanılacak")
                # Varsayılan intent etiketleri
                intent_labels = [
                    'nedir_ve_ne_icin_kullanilir',
                    'uygun_kullanim_doz_siklik', 
                    'olasi_yan_etkiler',
                    'hamilelikte_kullanim',
                    'emzirirken_kullanim',
                    'etkilesimler',
                    'saklama_ve_muhafaza',
                    'kullanilmamasi_gereken_durumlar',
                    'dikkatli_kullanim_gerektiren_durumlar'
                ]
            
            # Her şey başarılıysa kaydet
            self.bert_model = model
            self.bert_tokenizer = tokenizer
            self.intent_labels = intent_labels
            self.bert_enabled = True
            
            print(f"✅ BERT model başarıyla yüklendi")
            print(f"📋 Intent sayısı: {len(self.intent_labels)}")
            print(f"📊 Model parametreleri: {sum(p.numel() for p in self.bert_model.parameters()):,}")
            print(f"📊 Final vocab size: {self.bert_tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"❌ BERT model yükleme hatası: {e}")
            print("💡 Sadece Context-based mod ile devam ediliyor")
            self.bert_enabled = False
  
    def predict_intent_with_bert(self, question: str) -> Tuple[str, float]:
        """BERT ile intent tahmin eder"""
        if not self.bert_enabled:
            return "nedir_ve_ne_icin_kullanilir", 0.5
        
        try:
            # Metni tokenize et
            inputs = self.bert_tokenizer(
                question,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # GPU'ya taşı eğer varsa
            if torch.cuda.is_available() and next(self.bert_model.parameters()).is_cuda:
                inputs = {key: val.cuda() for key, val in inputs.items()}
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions.max().item()
            
            # Intent label'ını al
            if predicted_class_id < len(self.intent_labels):
                predicted_intent = self.intent_labels[predicted_class_id]
            else:
                predicted_intent = "nedir_ve_ne_icin_kullanilir"
            
            print(f"🤖 BERT Tahmin: {predicted_intent} (güven: {confidence:.3f})")
            
            return predicted_intent, confidence
            
        except Exception as e:
            print(f"⚠️ BERT tahmin hatası: {e}")
            return "nedir_ve_ne_icin_kullanilir", 0.5
    
    def find_drug_by_name(self, question: str) -> Tuple[Optional[Dict], str]:
        """Soruda geçen ilaç adını bulur ve ilgili veriyi döndürür"""
        question_lower = question.lower().strip()
        
        print(f"🔍 Debug: Aranan soru: '{question_lower}'")
        
        # En iyi eşleşmeyi bul
        best_match = None
        best_score = 0
        best_drug_name = ""
        
        # 1. ÖNCE: Tam eşleşme ara
        for drug_name in self.ilac_data.keys():
            if drug_name in question_lower or question_lower in drug_name:
                if len(drug_name) > best_score:
                    best_match = self.ilac_data[drug_name]
                    best_score = len(drug_name)
                    best_drug_name = drug_name
                    print(f"✅ Debug: Tam eşleşme bulundu: '{drug_name}'")
        
        # 2. SONRA: Kelime bazlı eşleşme
        if not best_match:
            question_words = question_lower.split()
            
            for drug_name in self.ilac_data.keys():
                drug_words = drug_name.split()
                
                # Her kelime için kontrol
                for q_word in question_words:
                    if len(q_word) >= 3:  # En az 3 karakter
                        for d_word in drug_words:
                            # Başlangıç eşleşmesi veya içerme
                            if (d_word.startswith(q_word) or q_word.startswith(d_word) or 
                                q_word in d_word or d_word in q_word):
                                score = len(q_word) + len(d_word)
                                if score > best_score:
                                    best_match = self.ilac_data[drug_name]
                                    best_score = score
                                    best_drug_name = drug_name
                                    print(f"✅ Debug: Kelime eşleşmesi: '{q_word}' -> '{d_word}' -> '{drug_name}'")
        
        # 3. SON ÇARE: Benzerlik bazlı eşleşme
        if not best_match:
            for drug_name in self.ilac_data.keys():
                # Levenshtein benzeri basit mesafe hesapla
                similarity = self._calculate_similarity(question_lower, drug_name)
                if similarity > 0.6 and similarity > best_score:
                    best_match = self.ilac_data[drug_name]
                    best_score = similarity
                    best_drug_name = drug_name
                    print(f"✅ Debug: Benzerlik eşleşmesi: '{drug_name}' (skor: {similarity:.2f})")
        
        if best_match:
            print(f"🎯 Debug: Seçilen ilaç: '{best_drug_name}' (skor: {best_score})")
            return best_match, best_drug_name
        else:
            print("❌ Debug: Hiç ilaç eşleşmedi")
            return None, ""
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """İki metin arasındaki benzerliği hesaplar (basit versiyon)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_answer(self, question):
        """Ana hibrit soru-cevap fonksiyonu - SADECE BERT + Context-based"""
        logger.info(f"🤔 Hibrit analiz başlıyor: {question}")
        
        if not question.strip():
            return {
                'answer': 'Bu ilaç hakkında bilgi bulunamadı.',
                'confidence': 0.0,
                'detected_drug': 'bulunamadı',
                'model_version': self.model_version,
                'method': 'Hybrid',
                'bert_intent': None,
                'bert_confidence': 0.0
            }
        
        # 1. İlaç adını bul
        drug_data, drug_name = self.find_drug_by_name(question)
        
        if not drug_data:
            # İlaç bulunamadı - basit mesaj dön
            return {
                'answer': 'Bu ilaç hakkında bilgi bulunamadı.',
                'confidence': 0.0,
                'detected_drug': 'bulunamadı',
                'model_version': self.model_version,
                'method': 'Hybrid',
                'bert_intent': None,
                'bert_confidence': 0.0
            }
        
        # 2. BERT ile intent tahmin et
        bert_intent, bert_confidence = self.predict_intent_with_bert(question)
        
        # 3. Context-based ile relevant context bul
        context_info = self.find_relevant_context_for_drug(question, drug_data, bert_intent, bert_confidence)
        
        if not context_info['context'] or len(context_info['context'].strip()) < 10:
            return {
                'answer': 'Bu ilaç hakkında bilgi bulunamadı.',
                'confidence': 0.0,
                'detected_drug': drug_name,
                'model_version': self.model_version,
                'method': 'Hybrid',
                'bert_intent': bert_intent,
                'bert_confidence': bert_confidence
            }
        
        # 4. Context'ten akıllı cevap üret
        try:
            answer = self._generate_smart_answer(question, context_info)
            
            # BERT güvenini hesaba katarak genel güven seviyesi belirle
            overall_confidence = self._calculate_overall_confidence(bert_confidence, answer, context_info)
            
            # Method belirleme
            method = 'BERT+Context' if self.bert_enabled and bert_confidence >= 0.8 else 'Context-Based'
            
            return {
                'answer': answer,
                'confidence': overall_confidence,
                'detected_drug': drug_name,
                'model_version': self.model_version,
                'method': method,
                'bert_intent': bert_intent,
                'bert_confidence': bert_confidence
            }
            
        except Exception as e:
            logger.error(f"⚠️ Cevap üretirken hata: {e}")
            # Hata durumunda da bilgi bulunamadı mesajı
            return {
                'answer': 'Bu ilaç hakkında bilgi bulunamadı.',
                'confidence': 0.0,
                'detected_drug': drug_name,
                'model_version': self.model_version,
                'method': 'Hybrid-Error',
                'bert_intent': bert_intent,
                'bert_confidence': bert_confidence
            }
    
    def find_relevant_context_for_drug(self, question: str, drug_data: Dict, bert_intent: str = None, bert_confidence: float = 0.5) -> Dict[str, str]:
        """Belirtilen ilaç için en uygun context'i bulur - 0.8 threshold ile optimized"""
        question_lower = question.lower()
        
        print(f"🔍 Debug: İlaç verisi anahtarları: {list(drug_data.keys())}")
        
        # Soru tipine göre uygun bölümü seç
        context = ''
        source = ''
        
        # ÖNCELİK 1: BERT Intent'i kullan - SADECE güven >= 0.8 ise
        if self.bert_enabled and bert_intent and bert_confidence >= 0.8:
            context = drug_data.get(bert_intent, '')
            if context and len(context.strip()) > 10:
                source = f'bert_{bert_intent}'
                print(f"🎯 Debug: BERT Intent bölümü seçildi (güven: {bert_confidence:.3f}): {bert_intent}")
            else:
                print(f"⚠️ Debug: BERT Intent bölümü boş (güven: {bert_confidence:.3f}): {bert_intent}")
        else:
            if self.bert_enabled and bert_intent:
                print(f"🔽 Debug: BERT confidence düşük ({bert_confidence:.3f} < 0.8), Context-based kullanılacak")
            
        # ÖNCELİK 2: Manuel kural bazlı (BERT başarısızsa veya düşük güven < 0.8)
        if not context or len(context.strip()) < 10:
            print("🔄 Debug: Context-based yaklaşım kullanılıyor...")
            
            if any(word in question_lower for word in ['hamile', 'gebelik', 'gebe']):
                context = drug_data.get('hamilelikte_kullanim', '')
                source = 'context_hamilelik'
                print("🎯 Debug: Hamilelik bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['emzir', 'anne', 'süt']):
                context = drug_data.get('emzirirken_kullanim', '')
                source = 'context_emzirme' 
                print("🎯 Debug: Emzirme bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['yan etki', 'zararlı', 'etki', 'problem', 'risk']):
                context = drug_data.get('olasi_yan_etkiler', '')
                source = 'context_yan_etkiler'
                print("🎯 Debug: Yan etkiler bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['doz', 'nasıl', 'ne kadar', 'kaç', 'miktar', 'alınır', 'kullanım']):
                # Önce uygun_kullanim_doz_siklik deneyelim, yoksa alternatifler
                context = drug_data.get('uygun_kullanim_doz_siklik', '') or drug_data.get('nedir_ve_ne_icin_kullanilir', '')
                source = 'context_doz_bilgisi'
                print("🎯 Debug: Doz bilgisi bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['etkileşim', 'beraber', 'birlikte']) and 'ilaç' in question_lower:
                if 'etkilesimler' in drug_data and drug_data['etkilesimler']:
                    context = self.format_etkilesimler(drug_data['etkilesimler'])
                    source = 'context_etkileşimler'
                    print("🎯 Debug: Etkileşimler bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['sakla', 'muhafaza', 'koruma', 'saklama']):
                context = drug_data.get('saklama_ve_muhafaza', '')
                source = 'context_saklama'
                print("🎯 Debug: Saklama bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['alkol', 'içki', 'alkollü']) and any(word in question_lower for word in ['beraber', 'birlikte', 'içilir']):
                # Alkol etkileşimi için özel kontrol
                context = drug_data.get('etkilesimler', '') or drug_data.get('dikkatli_kullanim_gerektiren_durumlar', '')
                source = 'context_alkol_etkileşim'
                print("🎯 Debug: Alkol etkileşimi bölümü seçildi (Context-based)")
            elif any(word in question_lower for word in ['nedir', 'ne için', 'kullanım', 'kullanılır', 'hastalık', 'tedavi']):
                context = drug_data.get('nedir_ve_ne_icin_kullanilir', '')
                source = 'context_kullanım_alanı'
                print("🎯 Debug: Kullanım alanı bölümü seçildi (Context-based)")
            else:
                # Varsayılan: kullanım alanı
                context = drug_data.get('nedir_ve_ne_icin_kullanilir', '')
                source = 'context_genel_bilgi'
                print("🎯 Debug: Genel bilgi bölümü seçildi (Context-based)")
        
        # Eğer seçilen bölüm boşsa alternatif ara - SADECE MEVCUT VERİDEN
        if not context or len(context.strip()) < 10:
            print("⚠️ Debug: Seçilen bölüm boş, alternatif aranıyor...")
            alternativeler = [
                ('nedir_ve_ne_icin_kullanilir', 'context_kullanım_alanı'),
                ('uygun_kullanim_doz_siklik', 'context_doz_bilgisi'),
                ('olasi_yan_etkiler', 'context_yan_etkiler'),
                ('hamilelikte_kullanim', 'context_hamilelik'),
                ('emzirirken_kullanim', 'context_emzirme'),
                ('saklama_ve_muhafaza', 'context_saklama')
            ]
            
            for alt_key, alt_source in alternativeler:
                alt_context = drug_data.get(alt_key, '')
                if alt_context and len(alt_context.strip()) > 10:
                    context = alt_context
                    source = alt_source
                    print(f"✅ Debug: Alternatif bölüm bulundu: {alt_key}")
                    break
        
        print(f"🎯 Debug: Final context uzunluk: {len(context)} karakter")
        
        return {
            'context': context,
            'source': source
        }
    

    
    def _generate_smart_answer(self, question: str, context_info: Dict) -> str:
        """Context'ten akıllı cevap üretir - SADECE mevcut context'ten"""
        context = context_info.get('context', '')
        
        if not context or len(context.strip()) < 10:
            return 'Bu ilaç hakkında bilgi bulunamadı.'
        
        # Soruya göre context'i akıllıca düzenle
        question_lower = question.lower()
        
        # Kısaltma ve formatla
        if len(context) > 400:
            # Uzun context'i akıllıca kısalt
            sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in sentences[:3]:  # İlk 3 cümle
                if sentence.strip():
                    relevant_sentences.append(sentence.strip())
            
            context = '. '.join(relevant_sentences) + '.'
        
        # Özel formatlamalar - SADECE mevcut context'e dayalı
        if 'alkol' in question_lower:
            if 'alkol' not in context.lower():
                # Alkol bilgisi yoksa context'i olduğu gibi döndür
                pass
        
        return context.strip()
    
    def _calculate_overall_confidence(self, bert_confidence: float, answer: str, context_info: Dict) -> float:
        """Genel güven seviyesini hesaplar"""
        if not answer or answer == 'Bu ilaç hakkında bilgi bulunamadı.':
            return 0.0
            
        base_confidence = 0.6
        
        # BERT confidence'ı hesaba kat (sadece yüksek güvenlerde)
        if self.bert_enabled and bert_confidence >= 0.8:
            base_confidence += (bert_confidence - 0.8) * 0.2
        
        # Cevap uzunluk kontrolü
        if len(answer) > 50:
            base_confidence += 0.1
        elif len(answer) < 20:
            base_confidence -= 0.1
        
        # Context source'a göre güven ayarlama
        source = context_info.get('source', '')
        if source.startswith('bert_'):
            base_confidence += 0.1
        elif source.startswith('context_'):
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)


# Model instance - global değişken olarak tanımlayalım
model = None

# In-memory storage (gerçek projede veritabanı kullanın)
feedback_storage = []
conversation_storage = []

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/Logo/<path:filename>')
def serve_logo(filename):
    """Logo dosyalarını serve et"""
    import os
    logo_path = os.path.join('Logo', filename)
    if os.path.exists(logo_path):
        from flask import send_file
        return send_file(logo_path)
    else:
        return "Logo bulunamadı", 404

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Soru sorma endpoint'i"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        user_id = data.get('user_id', 'anonymous')
        
        if not question:
            return jsonify({'error': 'Soru boş olamaz'}), 400
        
        # Model'den cevap al
        model_response = model.get_answer(question)
        
        # Conversation kaydet
        conversation_id = str(uuid.uuid4())
        conversation = {
            'id': conversation_id,
            'question': question,
            'answer': model_response['answer'],
            'confidence': model_response['confidence'],
            'detected_drug': model_response.get('detected_drug', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'model_version': model_response['model_version'],
            'method': model_response.get('method', 'Hybrid'),
            'bert_intent': model_response.get('bert_intent'),
            'bert_confidence': model_response.get('bert_confidence', 0.0)
        }
        
        conversation_storage.append(conversation)
        
        return jsonify(conversation)
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Feedback gönderme endpoint'i"""
    try:
        data = request.get_json()
        
        # Feedback ID oluştur
        feedback_id = f"{datetime.now().strftime('%Y-%m-%d')}-{int(time.time())}"
        
        # Feedback verisini yapılandır
        feedback_data = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "user_id": data.get('user_id', 'anonymous'),
            "question": data.get('question', ''),
            "model_answer": data.get('model_answer', ''),
            "user_feedback": {
                "type": data.get('feedback_type', ''),
                "reason": data.get('reason', ''),
                "additional_comment": data.get('comment', '')
            },
            "context_info": {
                "model_version": "hybrid-bert-context-v1.0",
                "dataset_version": "drug_leaflet_hybrid_v1",
                "platform": "web",
                "detected_drug": data.get('detected_drug', 'unknown'),
                "confidence_score": data.get('confidence', 0.0)
            },
            "status": "pending_review" if data.get('feedback_type') == 'dislike' else "completed"
        }
        
        # Feedback'i kaydet
        feedback_storage.append(feedback_data)
        
        # JSON dosyasına kaydet
        save_feedback_to_file(feedback_data)
        
        return jsonify({
            'success': True, 
            'message': 'Geribildiriminiz kaydedildi',
            'feedback_id': feedback_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/list', methods=['GET'])
def list_feedback():
    """Feedback listesi (admin için)"""
    return jsonify(feedback_storage)

def save_feedback_to_file(feedback_data):
    """Feedback'i GeriBildirim.json dosyasına kaydet"""
    try:
        filename = "GeriBildirim.json"
        
        # Mevcut dosyayı oku
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []
        
        # Yeni feedback'i ekle
        existing_data.append(feedback_data)
        
        # Dosyaya yaz
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Feedback başarıyla {filename} dosyasına kaydedildi")
            
    except Exception as e:
        print(f"❌ Dosya kaydetme hatası: {e}")

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>İlaç Danışman AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        .chat-message { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .loading-dots { animation: pulse 1.5s infinite; }
        /* Capsule animation */
        .capsule-wrapper { display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .capsule { position: relative; width: 220px; height: 80px; margin-top: 8px; }
        .capsule-half { position: absolute; top: 0; width: 110px; height: 80px; border-radius: 9999px; box-shadow: 0 6px 16px rgba(0,0,0,0.15); }
        .capsule-left { left: 0; background: linear-gradient(90deg, #ef4444, #f87171); border-right: 3px solid #ffffff; }
        .capsule-right { right: 0; background: linear-gradient(90deg, #93c5fd, #60a5fa); border-left: 3px solid #ffffff; }
        @keyframes openLeft { to { transform: translateX(-140px) rotate(-20deg); opacity: 0.95; } }
        @keyframes openRight { to { transform: translateX(140px) rotate(20deg); opacity: 0.95; } }
        .open-left { animation: openLeft 0.8s forwards ease-out; }
        .open-right { animation: openRight 0.8s forwards ease-out; }
        .pill-answer { max-width: 48rem; }
        /* Left fixed sidebar */
        @media (min-width: 1024px) {
            .left-sidebar { 
                position: fixed; 
                left: 0; 
                top: 0; 
                bottom: 0; 
                width: 320px; 
                overflow-y: auto;
                z-index: 10;
                background: #f7f7f8;
                border-right: 1px solid #e5e5e5;
            }
            .main-content { 
                margin-left: 320px; 
                max-width: calc(100vw - 320px);
            }
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div id="app">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b">
            <div class="max-w-4xl mx-auto px-4 py-4">
                <div class="flex items-center justify-center">
                    <div class="w-16 h-16 flex items-center justify-center">
                        <img src="/Logo/FarmaBotLogo.png" alt="FarmaBot Logo" class="w-full h-full object-contain">
                    </div>
                </div>
            </div>
        </header>

        <!-- Sol Sabit Bilgilendirme Paneli -->
        <aside class="left-sidebar hidden lg:block p-4">
            <div class="mb-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-3">FarmaBot size hangi konularda bilgi sağlayabilir?</h3>
                <p class="text-sm text-gray-700 mb-2">İlaç Hakkında:</p>
                <ul class="text-sm text-gray-700 list-disc pl-5 space-y-1">
                    <li>Nedir ve ne için kullanılır?</li>
                    <li>Etkin maddeleri nelerdir?</li>
                    <li>Yardımcı maddeleri nelerdir?</li>
                    <li>Kullanılmaması gereken durumlar nelerdir?</li>
                    <li>Dikkatli kullanılması gereken durumlar nelerdir?</li>
                    <li>Hamilelikte kullanımı nasıl olmalıdır?</li>
                    <li>Emzirme döneminde kullanımı nasıl olmalıdır?</li>
                    <li>Araç ve makine kullanımına etkisi var mıdır?</li>
                    <li>İlaç hakkında önemli uyarılar nelerdir?</li>
                    <li>Bilinmesi gereken diğer önemli bilgiler nelerdir?</li>
                    <li>Fazla kullanım durumunda ne yapılmalıdır?</li>
                    <li>Kullanım unutulursa ne yapılmalıdır?</li>
                    <li>Tedavi kesildiğinde ortaya çıkabilecek etkiler nelerdir?</li>
                    <li>Yan etkileri nelerdir?</li>
                    <li>Nasıl saklanmalıdır?</li>
                    <li>Son kullanma tarihi nedir?</li>
                </ul>
                <div class="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <p class="text-sm text-amber-800">
                        <strong>NOT:</strong> FarmaBot prospektüste bulunan bilgilerle oluşturulan bir veriseti ile sorularınızı cevaplamaktadır. Eğer eksik veya hatalı bilgi varsa bize iletiniz. Farmakoloji uzmanlarımıza sorununuzu iletelim.
                    </p>
                </div>
            </div>
        </aside>

        <!-- Ana İçerik -->
        <main class="main-content max-w-4xl mx-auto px-4 py-8">
            <!-- Sohbet Alanı -->
            <div class="bg-white rounded-lg shadow-sm border mb-6">
                <div class="p-6">
                    <div id="chat-container">
                        <!-- Başlangıç Mesajı -->
                        <div id="welcome-message" class="text-center py-12">
                            <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <span class="text-3xl">🤖</span>
                            </div>
                            <h2 class="text-xl font-semibold text-gray-900 mb-2">
                                Nasıl yardımcı olabilirim?
                            </h2>
                        </div>
                        
                        <!-- Sohbet Mesajları -->
                        <div id="messages" class="space-y-6 max-h-96 overflow-y-auto hidden"></div>
                        
                        <!-- Loading -->
                        <div id="loading" class="hidden flex items-center justify-center py-8">
                            <div class="flex items-center space-x-3 text-blue-600">
                                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                                <span>AI analiz ediyor...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Soru Formu -->
            <div class="bg-white rounded-lg shadow-sm border p-6">
                <div class="space-y-4">
                    <label for="question" class="block text-sm font-medium text-gray-700">
                        Sorunuzu yazın
                    </label>
                    <div class="relative">
                        <textarea
                            id="question"
                            placeholder="Örnek: Parol ile alkol alınır mı?"
                            class="w-full p-4 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            rows="4"
                            maxlength="500"
                        ></textarea>
                        <div id="char-count" class="absolute bottom-3 right-3 text-xs text-gray-400">0/500</div>
                    </div>
                    
                    <button
                        id="submit-btn"
                        class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 hover:shadow-lg hover:scale-105 transform disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-all duration-200"
                    >
                        <span>📤</span>
                        <span>Soru Sor</span>
                    </button>
                    <div id="capsule-area" class="mt-4"></div>
                </div>
                
                <!-- Uyarı -->
                <div class="mt-4">
                    <p class="text-sm font-bold text-gray-800 text-center">
                        BU BİLGİLER SADECE REHBER NİTELİĞİNDEDİR. KESİN TANI VE TEDAVİ İÇİN MUTLAKA DOKTORUNUZA VEYA ECZACINIZA DANIŞIN.
                    </p>
                </div>
            </div>
        </main>
    </div>

    <!-- Feedback Modal -->
    <div id="feedback-modal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div class="bg-white rounded-lg max-w-md w-full">
            <div class="flex items-center justify-between p-6 border-b">
                <h3 class="text-lg font-semibold text-gray-900">Geri Bildirim</h3>
                <button id="close-feedback" class="text-gray-400 hover:text-gray-600">❌</button>
            </div>
            
            <div class="p-6 space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-3">
                        Neden yararlı değil?
                    </label>
                    <div class="space-y-2" id="feedback-reasons">
                        <label class="flex items-center">
                            <input type="radio" name="reason" value="wrong_answer" class="h-4 w-4 text-blue-600">
                            <span class="ml-2 text-sm text-gray-700">Cevap yanlış</span>
                        </label>
                        <label class="flex items-center">
                            <input type="radio" name="reason" value="incomplete" class="h-4 w-4 text-blue-600">
                            <span class="ml-2 text-sm text-gray-700">Cevap eksik</span>
                        </label>
                        <label class="flex items-center">
                            <input type="radio" name="reason" value="unclear" class="h-4 w-4 text-blue-600">
                            <span class="ml-2 text-sm text-gray-700">Anlaşılmaz</span>
                        </label>
                        <label class="flex items-center">
                            <input type="radio" name="reason" value="irrelevant" class="h-4 w-4 text-blue-600">
                            <span class="ml-2 text-sm text-gray-700">İlgisiz</span>
                        </label>
                        <label class="flex items-center">
                            <input type="radio" name="reason" value="other" class="h-4 w-4 text-blue-600">
                            <span class="ml-2 text-sm text-gray-700">Diğer</span>
                        </label>
                    </div>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">
                        Ek yorum (opsiyonel)
                    </label>
                    <textarea
                        id="feedback-comment"
                        placeholder="Görüşünüzü paylaşın..."
                        class="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        rows="3"
                    ></textarea>
                </div>
            </div>
            
            <div class="flex space-x-3 p-6 border-t">
                <button id="cancel-feedback" class="flex-1 py-2 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
                    İptal
                </button>
                <button id="submit-feedback" class="flex-1 py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300">
                    Gönder
                </button>
            </div>
        </div>
    </div>

    <!-- Bildirim -->
    <div id="notification" class="hidden fixed top-4 right-4 z-50">
        <div id="notification-content" class="text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2">
            <span id="notification-text"></span>
            <button id="close-notification" class="text-white hover:text-gray-200">❌</button>
        </div>
    </div>

    <script>
        // Global variables
        let conversations = [];
        let currentFeedbackData = {};

        // DOM elements
        const questionInput = document.getElementById('question');
        const submitBtn = document.getElementById('submit-btn');
        const messagesContainer = document.getElementById('messages');
        const welcomeMessage = document.getElementById('welcome-message');
        const loadingDiv = document.getElementById('loading');
        const charCount = document.getElementById('char-count');
        const feedbackModal = document.getElementById('feedback-modal');

        // Event listeners
        questionInput.addEventListener('input', updateCharCount);
        questionInput.addEventListener('keydown', handleKeyDown);
        submitBtn.addEventListener('click', submitQuestion);
        document.getElementById('close-feedback').addEventListener('click', closeFeedbackModal);
        document.getElementById('cancel-feedback').addEventListener('click', closeFeedbackModal);
        document.getElementById('submit-feedback').addEventListener('click', submitFeedback);
        document.getElementById('close-notification').addEventListener('click', closeNotification);

        function updateCharCount() {
            const length = questionInput.value.length;
            charCount.textContent = `${length}/500`;
            submitBtn.disabled = length === 0;
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuestion();
            }
        }

        async function submitQuestion() {
            const question = questionInput.value.trim();
            if (!question || submitBtn.disabled) return;

            // UI güncellemeleri
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span>⏳</span><span>Gönderiliyor...</span>';
            welcomeMessage.classList.add('hidden');
            messagesContainer.classList.remove('hidden');
            loadingDiv.classList.remove('hidden');

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, user_id: 'anonymous' })
                });

                const data = await response.json();
                
                if (response.ok) {
                    conversations.push(data);
                    displayConversation(data);
                    playCapsuleReveal(data.answer);
                    questionInput.value = '';
                    updateCharCount();
                } else {
                    showNotification(data.error || 'Bir hata oluştu', 'error');
                }
            } catch (error) {
                showNotification('Bağlantı hatası', 'error');
            } finally {
                loadingDiv.classList.add('hidden');
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<span>📤</span><span>Soru Sor</span>';
            }
        }

        function displayConversation(conv) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'space-y-4 chat-message';
            messageDiv.innerHTML = `
                <!-- Kullanıcı Sorusu (Sağ taraf) -->
                <div class="flex items-start space-x-3 justify-end">
                    <div class="flex-1 max-w-xs">
                        <div class="bg-gray-100 rounded-lg px-4 py-3">
                            <p class="text-gray-900">${conv.question}</p>
                        </div>
                        <div class="flex items-center space-x-2 mt-1 text-xs text-gray-500 justify-end">
                            <span>🕐</span>
                            <span>${formatTime(conv.timestamp)}</span>
                        </div>
                    </div>
                    <div class="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span class="text-gray-600">👤</span>
                    </div>
                </div>

                <!-- AI Cevabı (Sol taraf) -->
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span class="text-blue-600">🤖</span>
                    </div>
                    <div class="flex-1">
                        <div class="bg-blue-50 rounded-lg px-4 py-3">
                            <p class="text-gray-900">${conv.answer}</p>
                            <!-- Model info badge -->
                            <div class="mt-2 text-xs text-gray-500">
                                <span class="inline-block bg-gray-200 rounded px-2 py-1 mr-2">${conv.method || 'Hybrid'}</span>
                                <span class="inline-block bg-gray-200 rounded px-2 py-1">Güven: ${(conv.confidence * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                        
                        <!-- Feedback Butonları -->
                        <div class="flex items-center space-x-3 mt-3" id="feedback-${conv.id}">
                            <button onclick="handleFeedback('${conv.id}', 'like')" 
                                    class="flex items-center space-x-2 text-gray-500 hover:text-green-600 transition-colors">
                                <span>👍</span>
                                <span class="text-sm">Yararlı</span>
                            </button>
                            <button onclick="handleFeedback('${conv.id}', 'dislike')" 
                                    class="flex items-center space-x-2 text-gray-500 hover:text-red-600 transition-colors">
                                <span>👎</span>
                                <span class="text-sm">Yararlı değil</span>
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function formatTime(timestamp) {
            return new Date(timestamp).toLocaleTimeString('tr-TR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        function handleFeedback(conversationId, type) {
            const conversation = conversations.find(c => c.id === conversationId);
            
            if (type === 'like') {
                submitPositiveFeedback(conversation);
            } else {
                currentFeedbackData = {
                    conversation_id: conversationId,
                    conversation: conversation,
                    type: 'dislike'
                };
                feedbackModal.classList.remove('hidden');
            }
        }

        async function submitPositiveFeedback(conversation) {
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: conversation.question,
                        model_answer: conversation.answer,
                        feedback_type: 'like',
                        reason: 'helpful',
                        comment: '',
                        detected_drug: conversation.detected_drug || 'unknown',
                        confidence: conversation.confidence || 0.0
                    })
                });

                if (response.ok) {
                    updateFeedbackUI(conversation.id, 'like');
                    showNotification('Teşekkürler! Geribildiriminiz kaydedildi.', 'success');
                }
            } catch (error) {
                showNotification('Geribildirimi kaydederken hata oluştu.', 'error');
            }
        }

        async function submitFeedback() {
            const reason = document.querySelector('input[name="reason"]:checked')?.value;
            const comment = document.getElementById('feedback-comment').value;

            if (!reason) {
                showNotification('Lütfen bir neden seçin.', 'error');
                return;
            }

            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: currentFeedbackData.conversation.question,
                        model_answer: currentFeedbackData.conversation.answer,
                        feedback_type: 'dislike',
                        reason: reason,
                        comment: comment,
                        detected_drug: currentFeedbackData.conversation.detected_drug || 'unknown',
                        confidence: currentFeedbackData.conversation.confidence || 0.0
                    })
                });

                if (response.ok) {
                    updateFeedbackUI(currentFeedbackData.conversation_id, 'dislike');
                    closeFeedbackModal();
                    showNotification('Geribildiriminiz kaydedildi. İncelemeye alınacak.', 'success');
                }
            } catch (error) {
                showNotification('Geribildirimi kaydederken hata oluştu.', 'error');
            }
        }

        function updateFeedbackUI(conversationId, type) {
            const feedbackDiv = document.getElementById(`feedback-${conversationId}`);
            feedbackDiv.innerHTML = `
                <div class="flex items-center space-x-2 text-green-600">
                    <span>✅</span>
                    <span class="text-sm">Geri bildiriminiz alındı</span>
                </div>
            `;
        }

        function closeFeedbackModal() {
            feedbackModal.classList.add('hidden');
            document.querySelectorAll('input[name="reason"]').forEach(radio => radio.checked = false);
            document.getElementById('feedback-comment').value = '';
        }

        function showNotification(message, type = 'info') {
            const notification = document.getElementById('notification');
            const content = document.getElementById('notification-content');
            const text = document.getElementById('notification-text');

            const colors = {
                success: 'bg-green-500',
                error: 'bg-red-500',
                info: 'bg-blue-500'
            };

            content.className = `text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 ${colors[type]}`;
            text.textContent = message;
            notification.classList.remove('hidden');

            setTimeout(() => {
                notification.classList.add('hidden');
            }, 3000);
        }

        function closeNotification() {
            document.getElementById('notification').classList.add('hidden');
        }

        // Capsule reveal animation
        function playCapsuleReveal(answerText) {
            const area = document.getElementById('capsule-area');
            if (!area) return;
            area.innerHTML = `
                <div class="capsule-wrapper">
                    <div class="capsule">
                        <div class="capsule-half capsule-left"></div>
                        <div class="capsule-half capsule-right"></div>
                    </div>
                    <div id="pill-answer" class="pill-answer opacity-0 transition-opacity duration-500 bg-blue-50 border border-blue-200 text-gray-800 rounded-lg p-4 mt-4 shadow-sm"></div>
                </div>
            `;

            const left = area.querySelector('.capsule-left');
            const right = area.querySelector('.capsule-right');
            const answer = area.querySelector('#pill-answer');
            // trigger opening
            requestAnimationFrame(() => {
                left.classList.add('open-left');
                right.classList.add('open-right');
            });

            // reveal answer after animation
            setTimeout(() => {
                answer.textContent = answerText;
                answer.classList.remove('opacity-0');
            }, 850);
        }

        // Sayfa yüklendiğinde
        document.addEventListener('DOMContentLoaded', function() {
            updateCharCount();
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 BERT + Context-Based İlaç Danışman AI başlatılıyor...")
    print("🤖 Sadece BERT ve Context-based hibrit sistem")
    print("📱 Web arayüzü: http://localhost:5000")
    print("🔧 API endpoints:")
    print("   POST /api/ask - Soru sorma")
    print("   POST /api/feedback - Feedback gönderme")
    print("   GET /api/feedback/list - Feedback listesi")
    print("\n⚠️  Model parametreleri:")
    print("   - İlaç veri dosyası: Zorunlu (örn: 'ilac_data.json')")
    print("   - BERT model yolu: Opsiyonel")
    print("   - Fallback/demo veri: KALDIRILDI")
    print("   - İlaç önerileri: KALDIRILDI")
    
    try:
        # Model başlatma - parametreleri burada ayarlayın
        data_path = r"C:\Users\adigu\Desktop\FarmaBot\Veri Seti\Dataset_.json"  # İlaç veri dosyanız - ZORUNLU
        bert_model_path = r"C:\Users\adigu\Desktop\FarmaBot\Model\bert_matching_model\best_model"
        
        print("🔄 Model başlatılıyor...")
        print(f"📂 Veri dosyası: {data_path}")
        
        model = HybridDrugAdvisorModel(data_path, bert_model_path)
        print("✅ Model başarıyla yüklendi!")
        
        # Flask uygulamasını başlat
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except FileNotFoundError as e:
        print(f"❌ Dosya bulunamadı: {e}")
        print("💡 İlaç veri dosyasının mevcut olduğundan emin olun")
        print("💡 Örnek dosya adı: 'ilac_data.json'")
    except Exception as e:
        print(f"❌ HATA: {e}")
        import traceback
        traceback.print_exc()