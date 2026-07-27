"""Create normalized CLIP embeddings for extracted keyframes."""

from stages import MODELS
from utils.model_output import release_models
from utils.records import frame_paths


def run(records, source, output, model_cache, force=False, offline=False):
    """Cache one normalized CLIP embedding for every available keyframe."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    cache = output / "cache" / "clip"
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        MODELS["clip"], cache_dir=str(model_cache), local_files_only=offline
    )
    model = CLIPModel.from_pretrained(
        MODELS["clip"], cache_dir=str(model_cache), local_files_only=offline
    ).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    for number, record in enumerate(records, 1):
        target = cache / (record["claim_id"] + ".npz")
        if target.exists() and not force:
            continue
        frames = frame_paths(source, record)
        if not frames:
            np.savez_compressed(target, embeddings=np.empty((0, 512)), frames=np.array([]))
            continue
        images = []
        for frame in frames:
            with Image.open(frame) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            features = features.pooler_output if hasattr(features, 'pooler_output') else features
        np.savez_compressed(
            target,
            embeddings=features.float().cpu().numpy(),
            frames=np.array([frame.name for frame in frames]),
        )
        if number % 25 == 0:
            print("clip", number, "/", len(records), flush=True)
    release_models(model, processor)
