from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── VADER ──────────────────────────────────────────────────────────────────────
vader = SentimentIntensityAnalyzer()

# ── RoBERTa (Cardiff NLP – Twitter-trained, 3-class: neg/neu/pos) ──────────────
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_pipeline = None
roberta_ready = False

def load_roberta():
    global roberta_pipeline, roberta_ready
    logger.info("Loading RoBERTa model …")
    roberta_pipeline = pipeline(
        "sentiment-analysis",
        model=ROBERTA_MODEL,
        tokenizer=ROBERTA_MODEL,
        truncation=True,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1,
    )
    roberta_ready = True
    logger.info("RoBERTa ready ✓")

# ── Helpers ────────────────────────────────────────────────────────────────────
def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "POSITIVE"
    elif compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"

def roberta_to_standard(label: str) -> str:
    """Normalise CardiffNLP labels → POSITIVE / NEUTRAL / NEGATIVE"""
    label = label.upper()
    mapping = {
        "POSITIVE": "POSITIVE",
        "LABEL_2":  "POSITIVE",
        "NEUTRAL":  "NEUTRAL",
        "LABEL_1":  "NEUTRAL",
        "NEGATIVE": "NEGATIVE",
        "LABEL_0":  "NEGATIVE",
    }
    return mapping.get(label, label)

def analyse(text: str) -> dict:
    # VADER
    t0 = time.perf_counter()
    vs = vader.polarity_scores(text)
    vader_ms = round((time.perf_counter() - t0) * 1000, 1)
    vader_result = {
        "label":    vader_label(vs["compound"]),
        "compound": round(vs["compound"], 4),
        "positive": round(vs["pos"], 4),
        "neutral":  round(vs["neu"], 4),
        "negative": round(vs["neg"], 4),
        "latency_ms": vader_ms,
    }

    # RoBERTa
    roberta_result = {"ready": False}
    if roberta_ready:
        t0 = time.perf_counter()
        raw = roberta_pipeline(text)[0]
        rob_ms = round((time.perf_counter() - t0) * 1000, 1)
        roberta_result = {
            "ready":      True,
            "label":      roberta_to_standard(raw["label"]),
            "confidence": round(raw["score"], 4),
            "latency_ms": rob_ms,
        }

    # Ensemble (simple majority / confidence-weighted when both available)
    if roberta_result["ready"]:
        ensemble_label = (
            roberta_result["label"]
            if roberta_result["confidence"] >= 0.6
            else vader_result["label"]
        )
    else:
        ensemble_label = vader_result["label"]

    return {
        "text":     text,
        "vader":    vader_result,
        "roberta":  roberta_result,
        "ensemble": ensemble_label,
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "vader_ready":   True,
        "roberta_ready": roberta_ready,
        "device":        "cuda" if torch.cuda.is_available() else "cpu",
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long (max 5 000 chars)"}), 400

    try:
        result = analyse(text)
        return jsonify(result)
    except Exception as e:
        logger.exception("Analysis error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch", methods=["POST"])
def batch():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Provide a non-empty 'texts' list"}), 400
    if len(texts) > 20:
        return jsonify({"error": "Max 20 texts per batch"}), 400

    results = []
    for t in texts:
        t = (t or "").strip()
        if t:
            results.append(analyse(t))
    return jsonify({"results": results, "count": len(results)})

# ── Boot ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading
    threading.Thread(target=load_roberta, daemon=True).start()
    app.run(debug=False, host="0.0.0.0", port=5000)
