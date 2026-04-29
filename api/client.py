import torch # type: ignore
from transformers import pipeline # type: ignore
import warnings

warnings.filterwarnings('ignore')

# Cache for loaded pipelines to avoid reloading
_pipeline_cache = {}

def get_summarization_pipeline(model_name):
    """Get or create summarization pipeline for model"""
    if model_name not in _pipeline_cache:
        try:
            # Use GPU if available, otherwise CPU
            device = 0 if torch.cuda.is_available() else -1
            
            print(f"Loading model: {model_name}...")
            
            _pipeline_cache[model_name] = pipeline(
                "summarization",
                model=model_name,
                device=device,
                trust_remote_code=True
            )
            
            print(f"Model {model_name} loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {str(e)}")
    
    return _pipeline_cache[model_name]

def call_summarization(text, model_name, max_length=120, min_length=30, prefix=""):
    """Call summarization using local model"""
    try:
        # Add prefix if needed (e.g., for T5)
        input_text = prefix + text if prefix else text
        
        # Truncate very long texts
        if len(input_text.split()) > 1024:
            words = input_text.split()
            input_text = " ".join(words[:1024])
            print("⚠️ Text truncated to 1024 words for processing")
        
        # Get pipeline
        summarizer = get_summarization_pipeline(model_name)
        
        # Summarize
        result = summarizer(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return result
        
    except Exception as e:
        return {"error": str(e)}