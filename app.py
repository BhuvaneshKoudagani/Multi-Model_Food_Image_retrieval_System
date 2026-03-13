"""
Flask Backend API for Multimodal Food Retrieval System
Run: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import io
import base64
import numpy as np
import torch
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "dataset"
CACHE_FILE    = "embeddings.npy"
PATHS_CACHE   = "image_paths.npy"
BATCH_SIZE    = 32
IMG_SIZE      = 128
TOP_K         = 5
CLIP_MODEL    = "openai/clip-vit-base-patch32"
FLUX_API_URL  = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
HF_TOKEN      = os.getenv("HF_TOKEN", "")
GENERATED_DIR = "generated_images"
os.makedirs(GENERATED_DIR, exist_ok=True)

# ── Load CLIP ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {device}...")
model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model.eval()
print("✅ CLIP ready")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_features(raw) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        feat = raw
    elif hasattr(raw, "image_embeds"):
        feat = raw.image_embeds
    elif hasattr(raw, "text_embeds"):
        feat = raw.text_embeds
    elif hasattr(raw, "pooler_output"):
        feat = raw.pooler_output
    elif hasattr(raw, "last_hidden_state"):
        feat = raw.last_hidden_state.mean(dim=1)
    else:
        feat = raw[0]
    while feat.dim() > 2:
        feat = feat.squeeze(1)
    if feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return feat


def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def build_or_load_index():
    if os.path.exists(CACHE_FILE) and os.path.exists(PATHS_CACHE):
        embs  = np.load(CACHE_FILE)
        paths = np.load(PATHS_CACHE, allow_pickle=True).tolist()
        embs  = embs.reshape(len(paths), -1)
        print(f"✅ Loaded {len(paths)} cached embeddings (dim={embs.shape[1]})")
        return embs, paths

    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(DATASET_PATH)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not paths:
        print(f"❌ No images in '{DATASET_PATH}'")
        sys.exit(1)

    print(f"Indexing {len(paths)} images...")
    all_embs = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [
            Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            for p in paths[start:start+BATCH_SIZE]
        ]
        inp = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = _safe_features(model.get_image_features(**inp))
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_embs.append(feat.cpu().numpy())
        print(f"  [{min(start+BATCH_SIZE,len(paths))}/{len(paths)}]", end="\r")

    print()
    embs = np.vstack(all_embs)
    np.save(CACHE_FILE, embs)
    np.save(PATHS_CACHE, np.array(paths, dtype=object))
    print(f"✅ Index built: {embs.shape}")
    return embs, paths


def embed_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    inp = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = _safe_features(model.get_image_features(**inp))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().reshape(1, -1)


def embed_text(text: str) -> np.ndarray:
    inp = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feat = _safe_features(model.get_text_features(**inp))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().reshape(1, -1)


def do_retrieve(query_emb: np.ndarray, top_k=TOP_K):
    q  = query_emb.reshape(1, -1)
    db = dataset_embs.reshape(len(dataset_paths), -1)
    if q.shape[1] != db.shape[1]:
        raise ValueError(f"Dim mismatch query={q.shape[1]} index={db.shape[1]}")
    scores  = cosine_similarity(q, db)[0]
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        img   = Image.open(dataset_paths[idx]).convert("RGB")
        cat   = os.path.basename(os.path.dirname(dataset_paths[idx]))
        results.append({
            "rank"     : int(top_idx.tolist().index(idx)) + 1,
            "score"    : round(float(scores[idx]), 4),
            "category" : cat,
            "image_b64": pil_to_b64(img)
        })
    return results


# ── Load index at startup ─────────────────────────────────────────────────────
dataset_embs, dataset_paths = build_or_load_index()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/retrieve/image", methods=["POST"])
def retrieve_by_image():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400
        img      = Image.open(file.stream).convert("RGB")
        q_emb    = embed_image(img)
        results  = do_retrieve(q_emb)
        query_b64 = pil_to_b64(img)
        return jsonify({"query_image": query_b64, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrieve/text", methods=["POST"])
def retrieve_by_text():
    try:
        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        q_emb   = embed_text(text)
        results = do_retrieve(q_emb)
        return jsonify({"query_text": text, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate_and_retrieve():
    try:
        data      = request.get_json()
        food_name = (data or {}).get("food_name", "").strip()
        if not food_name:
            return jsonify({"error": "No food name provided"}), 400
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set in .env"}), 400

        prompt = (
            f"ultra-realistic professional food photography of {food_name}, "
            "soft studio lighting, shallow depth of field, clean white plate, "
            "garnished beautifully, appetizing, 4k, award-winning"
        )
        resp = requests.post(
            FLUX_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
            timeout=90
        )
        if resp.status_code == 503:
            return jsonify({"error": "Model loading, wait 20s and retry"}), 503
        if resp.status_code != 200:
            return jsonify({"error": f"FLUX API error {resp.status_code}"}), 500

        gen_img   = Image.open(io.BytesIO(resp.content)).convert("RGB")
        fname     = os.path.join(GENERATED_DIR, f"{food_name.replace(' ','_')}.png")
        gen_img.save(fname)

        q_emb   = embed_image(gen_img)
        results = do_retrieve(q_emb)
        return jsonify({
            "generated_image": pil_to_b64(gen_img),
            "food_name"      : food_name,
            "results"        : results
        })
    except requests.exceptions.Timeout:
        return jsonify({"error": "FLUX API timed out, try again"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    return jsonify({
        "status"        : "ok",
        "device"        : device,
        "indexed_images": len(dataset_paths),
        "hf_token_set"  : bool(HF_TOKEN)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)