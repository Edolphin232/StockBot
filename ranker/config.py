import os
from dotenv import load_dotenv
import torch

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These anchors help create a "sentiment direction"
POSITIVE_TEXT = "The stock is expected to rise significantly"
NEGATIVE_TEXT = "The stock is expected to fall sharply"

# Finnhub configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
# Default lookback window for company news (in days)
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
# Maximum number of articles to consider per ticker
NEWS_MAX_ARTICLES = int(os.getenv("NEWS_MAX_ARTICLES", "50"))