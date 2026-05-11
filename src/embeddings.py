from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        normalized_texts = [text if text is not None else "" for text in texts]
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings.astype(np.float32))
