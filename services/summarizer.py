import torch  # type: ignore
from transformers import pipeline  # type: ignore
import warnings

warnings.filterwarnings("ignore")

# =========================
# CACHE MODEL
# =========================
_pipeline_cache = {}

def get_summarization_pipeline(model_name):
    """Load model sekali saja (cache)"""
    if model_name not in _pipeline_cache:
        try:
            device = 0 if torch.cuda.is_available() else -1

            summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device
            )

            _pipeline_cache[model_name] = summarizer

        except Exception as e:
            raise RuntimeError(f"Gagal load model {model_name}: {str(e)}")

    return _pipeline_cache[model_name]

# =========================
# CORE SUMMARIZATION
# =========================
def run_summarization(text, model_name, max_length=120, min_length=30, prefix=""):
    try:
        # Tambah prefix (T5)
        input_text = f"{prefix}{text}" if prefix else text

        # Batasi panjang teks
        words = input_text.split()
        if len(words) > 512:
            input_text = " ".join(words[:512])

        summarizer = get_summarization_pipeline(model_name)

        result = summarizer(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return result[0]["summary_text"]

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# MODEL CONFIG
# =========================
MODEL_CONFIG = {
    "Indo AI": {
        "model_name": "cahya/bert2bert-indonesian-summarization",
        "max_length": 100,
        "min_length": 20,
        "prefix": ""
    },
    "T5 Small": {
        "model_name": "t5-small",
        "max_length": 80,
        "min_length": 20,
        "prefix": "summarize: "
    }
}

# =========================
# WRAPPER FUNCTION
# =========================
def summarize_by_model(text, model_key):
    if model_key not in MODEL_CONFIG:
        return "❌ Model tidak ditemukan"

    config = MODEL_CONFIG[model_key]

    return run_summarization(
        text,
        model_name=config["model_name"],
        max_length=config["max_length"],
        min_length=config["min_length"],
        prefix=config["prefix"]
    )

# =========================
# SHORTCUT FUNCTIONS
# =========================
def summarize_indo(text):
    return summarize_by_model(text, "Indo AI")

def summarize_t5(text):
    return summarize_by_model(text, "T5 Small")

def summarize_simple(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    if not sentences:
        return "Teks kosong."
    if len(sentences) == 1:
        return sentences[0] + "."
    
    return ". ".join(sentences[:2]) + "."