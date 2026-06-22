#  AI Sentiment Analyzer

A dual-model sentiment analysis web app built with **Flask**, **VADER**, and **RoBERTa** — running fully locally with a clean browser-based UI.

> Internship Project — Built as a demonstration of NLP sentiment analysis using rule-based and transformer-based models with an ensemble decision layer.

---

##  Features

- **Dual Model Analysis** — VADER (rule-based, < 1ms) + RoBERTa (transformer, ~40ms)
- **Ensemble Decision** — combines both models for a final verdict
- **Single & Batch Mode** — analyze one text or up to 20 at once
- **Live Model Status** — real-time indicator as RoBERTa loads in the background
- **Session History** — last 10 analyses saved, click to replay
- **GPU Auto-detect** — uses CUDA if available, falls back to CPU
- **Keyboard Shortcut** — `Ctrl/Cmd + Enter` to analyze

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| Model 1 | VADER (`vaderSentiment`) |
| Model 2 | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| ML Framework | HuggingFace Transformers, PyTorch |
| Frontend | Vanilla HTML, CSS, JavaScript |

---

##  Project Structure

```
sentiment_analyzer/
├── app.py                  # Flask backend — routes, models, ensemble logic
├── requirements.txt        # Python dependencies
├── README.md
└── templates/
    └── index.html          # Frontend UI (served by Flask)
```

---

##  Setup & Installation

### Prerequisites
- Python 3.10+ (Anaconda recommended)
- ~600 MB free disk space (for RoBERTa model cache)
- Internet connection on first run (to download the model)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/0ByteBuilder1/-sentiment-analyzer.git
cd sentiment-analyzer
```

**2. Create and activate a conda environment**
```bash
conda create -n sentiment python=3.11
conda activate sentiment
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

> **First run:** RoBERTa (~500 MB) downloads automatically from HuggingFace.  
> **Subsequent runs:** loads from local cache in seconds.

---

##  API Reference

### `POST /api/analyze`
Analyze a single piece of text.

**Request:**
```json
{ "text": "This product is absolutely amazing!" }
```

**Response:**
```json
{
  "text": "This product is absolutely amazing!",
  "vader": {
    "label": "POSITIVE",
    "compound": 0.6249,
    "positive": 0.559,
    "neutral": 0.441,
    "negative": 0.0,
    "latency_ms": 0.4
  },
  "roberta": {
    "ready": true,
    "label": "POSITIVE",
    "confidence": 0.9821,
    "latency_ms": 38.2
  },
  "ensemble": "POSITIVE"
}
```

---

### `POST /api/batch`
Analyze multiple texts at once (max 20).

**Request:**
```json
{ "texts": ["Great experience!", "Terrible service.", "It was okay."] }
```

**Response:**
```json
{
  "count": 3,
  "results": [ ...same structure as /analyze, one per text... ]
}
```

---

### `GET /api/status`
Check if models are loaded.

**Response:**
```json
{
  "vader_ready": true,
  "roberta_ready": true,
  "device": "cpu"
}
```

---

##  How the Models Work

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Rule-based lexicon model — no training required
- Best for: short social media text, reviews, casual language
- Returns: `positive`, `neutral`, `negative` scores + a `compound` score (−1 to +1)
- Label thresholds: compound ≥ 0.05 → POSITIVE, ≤ −0.05 → NEGATIVE, else NEUTRAL

### RoBERTa (Cardiff NLP)
- Transformer model fine-tuned on ~58M tweets
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Returns: label + confidence score (0–1)

### Ensemble Logic
```
if RoBERTa confidence >= 0.60:
    use RoBERTa label
else:
    use VADER label
```

---

##  Notes

- The `HF_HUB_DISABLE_SYMLINKS_WARNING` warning on Windows is harmless — it just means the model cache uses copies instead of symlinks
- The `UNEXPECTED keys (pooler.dense)` log on startup is expected for this model architecture
- This is a development server — for production use, deploy with Gunicorn or uWSGI behind Nginx

---

## 📄 License

MIT License — free to use and modify.