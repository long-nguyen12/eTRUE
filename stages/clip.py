"""Create normalized CLIP embeddings for extracted keyframes."""

from stages import MODELS
from utils.model_output import release_models
from utils.records import frame_paths


def cached_embeddings_are_normalized(path, frame_names):
    """Return whether an existing cache contains finite unit-length rows."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as cached:
            embeddings = cached["embeddings"]
            cached_frames = [str(value) for value in cached["frames"]]
        if cached_frames != list(frame_names):
            return False
        if embeddings.ndim != 2:
            return False
        if embeddings.shape[0] == 0:
            return True
        norms = np.linalg.norm(embeddings, axis=1)
        return bool(np.isfinite(norms).all() and np.allclose(norms, 1.0, atol=1e-3))
    except (KeyError, OSError, ValueError):
        return False


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
        frames = frame_paths(source, record)
        frame_names = [frame.name for frame in frames]
        if (
            target.exists()
            and not force
            and cached_embeddings_are_normalized(target, frame_names)
        ):
            continue
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
            features = features.pooler_output if hasattr(features, "pooler_output") else features
            epsilon = torch.finfo(features.dtype).eps
            features = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(epsilon)
        np.savez_compressed(
            target,
            embeddings=features.float().cpu().numpy(),
            frames=np.array(frame_names),
        )
        if number % 25 == 0:
            print("clip", number, "/", len(records), flush=True)
    release_models(model, processor)
