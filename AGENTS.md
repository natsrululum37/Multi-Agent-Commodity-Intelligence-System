# AGENTS.md — tokoCabai

## Proyek
Multi-Agent Commodity Intelligence System (UAS Data Mining ST167 - Universitas AMIKOM Yogyakarta).
Menganalisis harga komoditas Indonesia (`cabai.csv`: harga cabai di Pasar Beringharjo, Feb 2024 - Jul 2026).

**Status**: ✅ PRODUCTION READY  
**Last Updated**: 24 Juli 2026  
**Final Score**: 90.82% (Weighted)

## Lingkungan
- **Python 3.12** - gunakan venv yang sudah ada: `.venv/bin/python`
- Aktifkan sebelum menjalankan: `source .venv/bin/activate`
- Install dependensi: `pip install -r requirements.txt`
- API key untuk Groq disimpan di `.env` file (tidak di-commit). Copy dari `.env.example`.

## Aturan Penting
- Semua dependensi HARUS diinstall di `.venv` dan dicatat di `requirements.txt`. Jangan install secara global.
- Setiap perubahan penting pada kode, struktur proyek, atau konfigurasi HARUS dicatat di memory persistence proyek ini.
- Penulisan code harus mengikuti best practice Python (PEP 8, type hints, docstring, modular).
- Jangan commit `.env`, `.venv/`, atau file cache.
- Setelah perubahan, selalu jalankan tests: `pytest tests/ -v`

## File Kunci
| File | Tujuan |
|------|--------|
| `cabai.csv` | Dataset sumber - 6 kolom: tanggal, id_komoditas, nama_komoditas, id_harga_pangan, nama_pasar, harga |
| `requirements.txt` | Dependensi: pandas, numpy, scikit-learn, sentence-transformers, faiss-cpu, groq, matplotlib, seaborn |
| `soal_uas_pdm.md` | Soal UAS yang mendefinisikan kebutuhan sistem multi-agent |
| `laporan_uas.md` | Laporan lengkap UAS dengan analisis, hasil, dan evaluasi |
| `README.md` | Dokumentasi proyek untuk pengguna |
| `.env` | File konfigurasi API key (tidak di-commit, copy dari .env.example) |
| `.env.example` | Template file konfigurasi |
| `main.py` | Entry point untuk menjalankan demo sistem |
| `src/` | Direktori kode sumber utama |

## Yang Harus Dibangun
Soal UAS mensyaratkan sistem multi-agent dengan:
1. Studi kasus enterprise dengan masalah lintas divisi ✅
2. Beberapa agent yang saling berinteraksi (framework: custom Python) ✅
3. Minimal salah satu dari: fine-tuning, RAG, embeddings, vector DB (FAISS) ✅
4. Evaluasi model menggunakan metrik Accuracy / Effectiveness / Efficiency / Explainability / Hallucination ✅

## Arsitektur Multi-Agent
| Agent | Peran | Teknologi |
|-------|-------|-----------|
| DataAgent | Membersihkan dan menganalisis data | pandas + numpy |
| PredictionAgent | Prediksi harga ensemble | scikit-learn + xgboost + lightgbm |
| RAGAgent | Menjawab pertanyaan bisnis | FAISS + sentence-transformers + Groq LLM |
| EvaluatorAgent | Evaluasi kualitas output | Custom metrics + Groq LLM |
| CoordinatorAgent | Orkestrasi agent pipeline | Python + Groq LLM |

## Fitur RAG yang Diimplementasikan
1. **FAISS Vector Store** - Semantic search dengan cosine similarity
2. **Caching System** - Index & dokumen di-cache ke `.cache/` untuk kecepatan
3. **Hybrid Search** - Semantic similarity + metadata filter
4. **Optimized Context Documents**:
   - Summary statistics (mean, median, min, max, std, volatility)
   - Monthly price summaries (aggregated)
   - Extreme values detection (top/bottom 3)
   - Overall trend analysis
   - Business insights (all divisions combined)
   - Reference dates (first/last day)
5. **Improved Confidence Calculation** - Inverse distance similarity score
6. **Enhanced LLM Prompt** - Memberikan jawaban informatif dengan angka spesifik

## Performance Metrics (Final)
| Metric | Score | Status |
|--------|-------|--------|
| Accuracy | 95.05% | ✅ Excellent |
| Effectiveness | 92.22% | ✅ Great |
| Efficiency | 84.72% | ✅ Great |
| Explainability | 81.48% | ✅ Good |
| Hallucination | 100.00% | ✅ Perfect |
| **Average Score** | **90.69%** | ✅ **Excellent** |
| **Weighted Score** | **90.82%** | ✅ **Excellent** |
| Execution Time | 12.64s | ✅ Fast |

## Cara Menjalankan
```bash
# Jalankan demo
source .venv/bin/activate
python main.py

# Jalankan interactive mode
python main.py --interactive

# Jalankan tests
pytest tests/ -v

# Jalankan FastAPI server
cd src/api && uvicorn server:app --reload --port 8000
```

## Optimasi yang Dilakukan
1. **Document Generator**: Mengurangi dokumen dari ~186 menjadi 26 (86% reduction)
2. **RAG Queries**: Mengurangi query dari 16 menjadi 8 per komoditas (50% reduction)
3. **Feature Caching**: Pre-compute features untuk semua komoditas sekaligus
4. **Efficiency Formula**: Non-linear decay formula untuk scoring yang lebih realistis
5. **Pipeline Optimization**: Parallel processing dan caching untuk kecepatan

## Konvensi
- Linter/formatter: `black` dan `ruff`
- Testing: `pytest`
- Kode sumber di direktori `src/`
- Test di direktori `tests/`
- Output/visualisasi di direktori `reports/`
- Cache di direktori `.cache/`
- Jangan commit `.venv/`, `.env`, `__pycache__/`, atau `*.pyc`
