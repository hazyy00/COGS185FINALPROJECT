"""CLIP score evaluation for generated images."""

import torch
import open_clip
from PIL import Image


_model = None
_preprocess = None
_tokenizer = None


def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
    return _model, _preprocess, _tokenizer


def compute_clip_score(image: Image.Image, prompt: str) -> float:
    """Compute CLIP similarity score between an image and a text prompt.

    Returns a score in [0, 1] where higher means better alignment.
    """
    model, preprocess, tokenizer = _load_model()

    image_tensor = preprocess(image).unsqueeze(0)
    text_tensor = tokenizer([prompt])

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()

    # Normalize from [-1, 1] cosine similarity to [0, 1]
    return (score + 1) / 2
