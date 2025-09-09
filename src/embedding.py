import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        logger.info(f"Initializing embedding model with {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: list[str], show_progress_bar: bool = True):
        # Implement embedding logic here
        embedding = self.model.encode(
            text,
            batch_size=64,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embedding.tolist()


class MarqoEmbedder:
    def __init__(self, model_name) -> None:
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

    def embed(self, text: list[str]):
        processed = self.processor(text=text, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_text_features(
                processed["input_ids"], normalize=True
            )
        return embedding.tolist()
