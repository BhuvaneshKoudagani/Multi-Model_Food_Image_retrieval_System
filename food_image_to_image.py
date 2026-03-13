"""
Fast Multi-Modal Food Image Retrieval System
Uses CLIP ViT-B/32 with key optimizations:
  - Batch processing  → encodes many images at once
  - NumPy disk cache  → skip re-encoding on repeat runs
  - Resized inputs    → faster preprocessing (128×128)
  - L2 normalisation  → pure dot-product similarity (no cosine overhead)
  - Loop input        → query multiple images without re-indexing
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
CACHE_FILE   = "embeddings.npy"
PATHS_CACHE  = "image_paths.npy"
BATCH_SIZE   = 32        # tune up if you have GPU, down if RAM is tight
IMG_SIZE     = 128       # smaller = faster; 128 keeps enough detail for food
TOP_K        = 5
MODEL_NAME   = "openai/clip-vit-base-patch32"
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading CLIP ViT model…")
model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


def open_resized(path: str) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def build_or_load_index():
    """Return (embeddings_array, image_paths_list), using disk cache when available."""
    if os.path.exists(CACHE_FILE) and os.path.exists(PATHS_CACHE):
        print("✅  Loading cached embeddings (delete .npy files to rebuild)…")
        embs  = np.load(CACHE_FILE)
        paths = np.load(PATHS_CACHE, allow_pickle=True).tolist()
        print(f"    {len(paths)} images in index.")
        return embs, paths

    paths = [
        os.path.join(r, f)
        for r, _, files in os.walk(DATASET_PATH)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not paths:
        raise FileNotFoundError(f"No images found under '{DATASET_PATH}'")

    print(f"Indexing {len(paths)} images (batch={BATCH_SIZE})…")
    all_embs = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch = [open_resized(p) for p in paths[start:start + BATCH_SIZE]]
        inp   = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inp)
            if not isinstance(feat, torch.Tensor):
                feat = feat.image_embeds if hasattr(feat, "image_embeds") else feat[0]
            feat = feat / feat.norm(dim=-1, keepdim=True)   # L2-normalise
        all_embs.append(feat.cpu().numpy())
        done = min(start + BATCH_SIZE, len(paths))
        print(f"  [{done}/{len(paths)}]", end="\r")

    print()
    embs = np.vstack(all_embs)
    np.save(CACHE_FILE, embs)
    np.save(PATHS_CACHE, np.array(paths, dtype=object))
    print("💾  Index cached to disk for future runs.")
    return embs, paths


def encode_query(path: str) -> np.ndarray:
    img = open_resized(path)
    inp = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inp)
        if not isinstance(feat, torch.Tensor):
            feat = feat.image_embeds if hasattr(feat, "image_embeds") else feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().reshape(1, -1)


def show_results(query_path: str, embs: np.ndarray, paths: list[str]):
    q_emb  = encode_query(query_path)
    embs_2d = embs.reshape(len(embs), -1)   # ensure (N, D)
    q_emb   = q_emb.reshape(1, -1)          # ensure (1, D)
    scores = cosine_similarity(q_emb, embs_2d)[0]
    top_idx = scores.argsort()[::-1][:TOP_K]

    print(f"\nTop-{TOP_K} results for: {query_path}")
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.5))
    fig.patch.set_facecolor("#111827")

    # Query panel
    axes[0].imshow(Image.open(query_path).convert("RGB"))
    axes[0].set_title("QUERY", color="#f97316", fontsize=9, fontweight="bold", pad=4)
    axes[0].axis("off")
    for sp in axes[0].spines.values():
        sp.set_edgecolor("#f97316"); sp.set_linewidth(2.5)

    rank_colors = ["#fbbf24", "#9ca3af", "#b45309", "#6b7280", "#6b7280"]

    for rank, idx in enumerate(top_idx):
        score = scores[idx]
        cat   = os.path.basename(os.path.dirname(paths[idx]))
        print(f"  #{rank+1}  {score:.4f}  {paths[idx]}")
        ax = axes[rank + 1]
        ax.imshow(Image.open(paths[idx]).convert("RGB"))
        ax.set_title(f"#{rank+1} · {score:.3f}\n{cat}", color="white", fontsize=7.5, pad=4)
        ax.axis("off")
        c = rank_colors[rank]
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2)

    plt.suptitle("🍕 Food Image Retrieval — CLIP ViT", color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    embs, paths = build_or_load_index()

    while True:
        qp = input("\nQuery image path (or 'q' to quit): ").strip().strip('"')
        if qp.lower() == "q":
            break
        if not os.path.isfile(qp):
            print("File not found — try again.")
            continue
        show_results(qp, embs, paths)