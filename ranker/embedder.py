from sentence_transformers import SentenceTransformer
from .config import MODEL_NAME, DEVICE

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True, device=DEVICE)
