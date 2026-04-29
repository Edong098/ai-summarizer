from dotenv import load_dotenv # type: ignore
import os

load_dotenv()

# Model configurations - using ONLY models that support summarization task
MODEL_CONFIG = {
    "T5 Small": {
        "model_name": "t5-small",
        "task": "summarization",
        "prefix": "summarize: ",
        "max_length": 600,
        "min_length": 30
    },
    "T5 Base": {
        "model_name": "t5-base",
        "task": "summarization",
        "prefix": "summarize: ",
        "max_length": 600,
        "min_length": 30
    },
    "BART CNN": {
        "model_name": "facebook/bart-large-cnn",
        "task": "summarization",
        "prefix": "",
        "max_length": 600,
        "min_length": 30
    },
    "mT5 Base": {
        "model_name": "google/mt5-base",
        "task": "summarization",
        "prefix": "summarize: ",
        "max_length": 600,
        "min_length": 30
    }
}

# Default model
DEFAULT_MODEL = "T5 Small"