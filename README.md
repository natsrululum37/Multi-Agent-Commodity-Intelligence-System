# 🌶️ Multi-Agent Commodity Intelligence System

**UAS Data Mining ST167 - Universitas AMIKOM Yogyakarta**

Sistem kecerdasan buatan berbasis multi-agent untuk menganalisis dan memprediksi harga komoditas cabai di Pasar Beringharjo, Yogyakarta.

## 📋 Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Struktur Proyek](#struktur-proyek)
- [Kontribusi](#kontribusi)

## ✨ Fitur Utama

### 1. **Multi-Agent Architecture**
Sistem terdiri dari 5 agent yang saling berinteraksi:
- **DataAgent**: Memuat, membersihkan, dan menganalisis data historis
- **PredictionAgent**: Memprediksi harga menggunakan machine learning
- **RAGAgent**: Menjawab pertanyaan bisnis dengan Retrieval-Augmented Generation
- **EvaluatorAgent**: Mengevaluasi kualitas output sistem
- **CoordinatorAgent**: Mengoordinasikan seluruh pipeline

### 2. **Machine Learning Prediction**
- Multi-model ensemble (Linear Regression, Random Forest, XGBoost, LightGBM)
- Feature engineering dengan lookback days
- Evaluasi performa dengan MAE, RMSE, R², MAPE
- Akurasi rata-rata: **95.05%**

### 3. **RAG System dengan FAISS**
- Semantic search menggunakan sentence-transformers
- Vector database FAISS untuk retrieval cepat
- Dokumentasi granular per komoditas
- Caching system untuk efisiensi

### 4. **Business Intelligence**
- Analisis tren harga (naik/turun/stabil)
- Rekomendasi per divisi (Procurement, Sales, Finance, Logistics, Risk Management)
- Volatilitas harga dan manajemen risiko
- Visualisasi data interaktif

### 5. **Evaluation Metrics**
Sistem evaluasi komprehensif dengan 5 metrik:
- ✅ **Accuracy**: 95.05% (Prediksi Harga)
- ✅ **Effectiveness**: 92.22% (RAG Responses)
- ✅ **Efficiency**: 84.72% (Waktu Eksekusi)
- ✅ **Explainability**: 81.48% (Penjelasan Bisnis)
- ✅ **Hallucination**: 100.00% (Kualitas Konten)

**Average Score: 90.69%** | **Weighted Score: 90.82%**

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    CoordinatorAgent                         │
│                   (Orchestration & LLM)                     │
└───────────���───┬─────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────────┬──────────────┐
    │           │           │               │              │
    ▼           ▼           ▼               ▼              ▼
┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐
│DataAgent│ │Prediction│ │ RAGAgent│ │Evaluator │ │ Document   │
│        │ │  Agent   │ │         │ │  Agent   │ │ Generator  │
└────────┘ └──────────┘ └─────────┘ └──────────┘ └────────────┘
    │           │           │               │              │
    ▼           ▼           ▼               ▼              ▼
  CSV Data   ML Models   FAISS Index    Metrics      Context Docs
```

### Detail Agent:

| Agent | Peran | Teknologi |
|-------|-------|-----------|
| **DataAgent** | Data loading, cleaning, analysis | pandas, numpy |
| **PredictionAgent** | Price prediction, trend analysis | scikit-learn, xgboost, lightgbm |
| **RAGAgent** | Business Q&A with context | FAISS, sentence-transformers, Groq LLM |
| **EvaluatorAgent** | System performance evaluation | Custom metrics, Groq LLM |
| **CoordinatorAgent** | Pipeline orchestration | Python, Groq LLM |

## 📦 Instalasi

### Prerequisites
- Python 3.12+
- pip package manager

### Langkah Instalasi

1. **Clone repository**
```bash
git clone https://github.com/natsrululum37/Multi-Agent-Commodity-Intelligence-System.git
cd Multi-Agent-Commodity-Intelligence-System
```

2. **Buat virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# atau
.venv\Scripts\activate  # Windows
```

3. **Install dependensi**
```bash
pip install -r requirements.txt
```

4. **Setup API key**
```bash
cp .env.example .env
# Edit .env dan masukkan GROQ_API_KEY Anda
```

5. **Jalankan test (opsional)**
```bash
pytest tests/ -v
```

## 🚀 Cara Penggunaan

### Quick Start

```bash
# Jalankan demo utama
python main.py

# Jalankan interactive mode (tanya jawab)
python main.py --interactive
```

### Output yang Dihasilkan

1. **Console Output**: Analisis lengkap per komoditas
2. **Reports** (`reports/`):
   - `price_comparison.png`: Perbandingan harga semua komoditas
   - `price_{commodity}.png`: Grafik harga per komoditas
   - `evaluation_metrics.png`: Dashboard evaluasi sistem

### Contoh Output

```
======================================================================
Multi-Agent Commodity Intelligence System
UAS Data Mining ST167 - Universitas AMIKOM Yogyakarta
======================================================================

📊 DataAgent: Memuat dan menganalisis data...
🔮 PredictionAgent: Memprediksi harga per komoditas...
🤖 RAGAgent: Membangun index dan menjawab pertanyaan...
📈 EvaluatorAgent: Mengevaluasi hasil...

📈 EVALUATION METRICS:
  ✅ Accuracy - Prediksi Harga: 95.05%
  ✅ Effectiveness - RAG Responses: 92.22%
  ✅ Efficiency - Waktu Eksekusi: 84.72%
  ✅ Explainability - Penjelasan Bisnis: 81.48%
  ✅ Hallucination - Kualitas Konten: 100.00%

  Average Score: 90.69%
  Weighted Score: 90.82%

⏱️  Execution Time: 12.64 seconds
```

## 📊 Hasil Evaluasi

### Metric Breakdown

| Metric | Weight | Score | Status |
|--------|--------|-------|--------|
| **Accuracy** | 30% | 95.05% | ✅ Excellent |
| **Effectiveness** | 25% | 92.22% | ✅ Great |
| **Efficiency** | 20% | 84.72% | ✅ Great |
| **Explainability** | 15% | 81.48% | ✅ Good |
| **Hallucination** | 10% | 100.00% | ✅ Perfect |

### Komoditas yang Dianalisis

| Komoditas | Avg Price | Volatility | Trend | R² Score |
|-----------|-----------|------------|-------|----------|
| Cabai Merah Besar | Rp 42,607 | 33.24% | ↓ Turun | 0.9724 |
| Cabai Merah Keriting | Rp 38,236 | 36.60% | → Mixed | 0.9741 |
| Cabai Rawit Hijau | Rp 44,456 | 30.79% | ↑ Naik | -5.4094 |
| Cabai Rawit Merah | Rp 48,856 | 45.31% | ↓ Turun | 0.9580 |

### Insight Utama

1. **Volatilitas Tinggi**: Semua komoditas menunjukkan volatilitas >30%, memerlukan strategi manajemen risiko
2. **Tren Umum**: 3 dari 4 komoditas menunjukkan tren penurunan harga
3. **Waktu Terbaik Beli**: Bulan Juli untuk semua komoditas (harga terendah)
4. **Prediksi Akurat**: Model mencapai R² >0.95 untuk 3 dari 4 komoditas

## 📁 Struktur Proyek

```
tokoCabai/
├── main.py                      # Entry point utama
├── requirements.txt             # Dependensi proyek
├── soal_uas_pdm.md             # Soal UAS
├── AGENTS.md                   # Dokumentasi agents
├── README.md                   # File ini
├── .env.example                # Template API key
├── .gitignore                  # Git ignore rules
├── cabai.csv                   # Dataset sumber
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Konfigurasi aplikasi
│   ├── agents/                 # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Base agent class
│   │   ├── coordinator.py      # Coordinator agent
│   │   ├── data_agent.py       # Data processing agent
│   │   ├── prediction_agent.py # ML prediction agent
│   │   ├── rag_agent.py        # RAG/Q&A agent
│   │   └── evaluator_agent.py  # Evaluation agent
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loader
│   │   └── preprocessing.py    # Data cleaning & features
│   ├── models/                 # ML models
│   │   ├── __init__.py
│   │   └── price_predictor.py  # Price prediction models
│   └── vectorstore/            # RAG components
│       ├── __init__.py
│       ├── document_generator.py # Context generation
│       └── faiss_index.py      # Vector store
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_agents.py          # Agent tests
��   ├── test_data.py            # Data tests
│   └── test_models.py          # Model tests
│
├── reports/                    # Generated outputs
│   ├── price_comparison.png
│   ├── price_cabai_merah_besar.png
│   ├── price_cabai_merah_keriting.png
│   ├── price_cabai_rawit_hijau.png
│   ├── price_cabai_rawit_merah.png
│   └── evaluation_metrics.png
│
├── .cache/                     # Cache directory
│   └── rag_index.pkl           # RAG index cache
│
└── .venv/                      # Virtual environment
```

## 🔧 Teknologi yang Digunakan

### Core Technologies
- **Python 3.12**: Bahasa pemrograman utama
- **pandas**: Data manipulation & analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **xgboost**: Gradient boosting
- **lightgbm**: Fast gradient boosting

### AI/ML Components
- **sentence-transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **Groq API**: Fast LLM inference (Llama 3.3 70B)

### Visualization
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Linter

## 📝 License

Proyek ini dibuat untuk keperluan UAS Data Mining ST167 di Universitas AMIKOM Yogyakarta.

## 👥 Tim Pengembang

**Proyek Data Mining - ST167**
- Universitas AMIKOM Yogyakarta
- Semester Genap 2025/2026

## 📞 Kontak

Untuk pertanyaan atau kontribusi, silakan buat issue di repository.

---

**Selamat Menggunakan!** 🌶️
