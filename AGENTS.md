# AGENTS.md — tokoCabai

## Proyek
Multi-Agent Commodity Intelligence System (UAS Data Mining ST167 - Universitas AMIKOM Yogyakarta).
Menganalisis harga komoditas Indonesia (`cabai.csv`: harga cabai merah di Pasar Beringharjo, Feb-Agu 2024).

## Lingkungan
- **Python 3.12** - gunakan venv yang sudah ada: `.venv/bin/python`
- Aktifkan sebelum menjalankan: `source .venv/bin/activate`
- Install dependensi: `pip install -r requirements.txt`

## Aturan Penting
- Semua dependensi HARUS diinstall di `.venv` dan dicatat di `requirements.txt`. Jangan install secara global.
- Setiap perubahan penting pada kode, struktur proyek, atau konfigurasi HARUS dicatat di memory persistence proyek ini.
- Penulisan code harus mengikuti best practice Python (PEP 8, type hints, docstring, modular).
- API key untuk Groq disimpan di `.env` file (tidak di-commit). Copy dari `.env.example`.

## File Kunci
| File | Tujuan |
|------|--------|
| `cabai.csv` | Dataset sumber - 6 kolom: tanggal, id_komoditas, nama_komoditas, id_harga_pangan, nama_pasar, harga |
| `requirements.txt` | Dependensi: pandas, numpy, scikit-learn, sentence-transformers, faiss-cpu, groq, matplotlib, seaborn |
| `soal_uas_pdm.md` | Soal UAS yang mendefinisikan kebutuhan sistem multi-agent (fine-tuning, RAG, embedding, vector DB, evaluasi) |
| `.env` | File konfigurasi API key (tidak di-commit, copy dari .env.example) |
| `.env.example` | Template file konfigurasi |
| `src/` | Direktori kode sumber utama |
| `main.py` | Entry point untuk menjalankan demo sistem |

## Yang Harus Dibangun
Soal UAS mensyaratkan sistem multi-agent dengan:
1. Studi kasus enterprise dengan masalah lintas divisi ✅
2. Beberapa agent yang saling berinteraksi (framework: LangChain, AutoGen, atau sejenisnya) ✅
3. Minimal salah satu dari: fine-tuning, RAG, embeddings, vector DB (FAISS) ✅
4. Evaluasi model menggunakan metrik Accuracy / Effectiveness / Efficiency / Explainability / Hallucination ✅

## Arsitektur Multi-Agent
| Agent | Peran | Teknologi |
|-------|-------|-----------|
| DataAgent | Membersihkan dan menganalisis data | pandas + scikit-learn |
| PredictionAgent | Prediksi harga | scikit-learn |
| RAGAgent | Menjawab pertanyaan bisnis | FAISS + sentence-transformers + Groq LLM |
| EvaluatorAgent | Evaluasi kualitas output | Custom metrics + Groq LLM |
| CoordinatorAgent | Orkestrasi agent | Python + Groq LLM |

## Fitur RAG yang Diimplementasikan
1. **FAISS Vector Store** - Semantic search dengan cosine similarity
2. **Caching System** - Index & dokumen di-cache ke `.cache/` untuk kecepatan
3. **Hybrid Search** - Semantic similarity + metadata filter (bulan, kuartal, tahunan)
4. **Context Documents Granular**:
   - Summary statistics (mean, median, min, max, std)
   - Monthly stats (rata-rata per bulan)
   - Quarterly distribution (Q1-Q4)
   - Year-over-year comparison
   - Volatility analysis (CV, standar deviasi)
   - Extreme values detection
   - Weekly trend analysis
   - Reference dates (first/last day)
5. **Improved Confidence Calculation** - Inverse distance similarity score
6. **Enhanced LLM Prompt** - Memberikan jawaban lebih informatif dengan angka spesifik

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

## Konvensi
- Linter/formatter: `black` dan `ruff`
- Testing: `pytest`
- Kode sumber di direktori `src/`
- Test di direktori `tests/`
- Output/visualisasi di direktori `reports/`
- Jangan commit `.venv/`, `.env`, atau `__pycache__/`
