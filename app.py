"""
Flask Backend API for Multimodal Food Retrieval System
Supports multiple datasets: Food-101 + Indian Food Dataset
Run: python app.py

Dataset structure expected:
  food-101/images/<category>/<image>.jpg
  indian-food/images/<category>/<image>.jpg

To re-index after adding the Indian food dataset:
  python app.py --reindex-indian
"""

import os
import sys
import io
import base64
import argparse
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

# ── Argument parsing (for CLI re-index flags) ─────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--reindex-indian", action="store_true",
                    help="Force re-index of the Indian food dataset and merge with existing cache")
parser.add_argument("--reindex-all", action="store_true",
                    help="Re-index ALL datasets from scratch (slow)")
cli_args, _ = parser.parse_known_args()

app = Flask(__name__, static_folder="static")
CORS(app, origins="*")

# ── Config ────────────────────────────────────────────────────────────────────
# Dataset paths — adjust these to your actual folder locations
DATASETS = {
    "food101": {
        "path":  "food-101/images",          # existing Food-101 images folder
        "label": "food101",
    },
    "indian": {
        "path":  "indian-food/images",        # new Indian food images folder
        "label": "indian",
    },
}

# Cache files — one combined cache storing embeddings + paths from ALL datasets
BASE_CACHE_DIR = os.path.expanduser("~/FoodRetrievalCache")
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

# Per-dataset cache (so you can add new datasets without re-indexing old ones)
DATASET_CACHE = {
    name: {
        "embs":  os.path.join(BASE_CACHE_DIR, f"{name}_embeddings.npy"),
        "paths": os.path.join(BASE_CACHE_DIR, f"{name}_paths.npy"),
    }
    for name in DATASETS
}

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


def index_dataset(name: str, ds_info: dict, force: bool = False):
    """
    Index a single dataset. Returns (embeddings np.ndarray, paths list).
    Uses cached version if available and force=False.
    """
    cache_embs  = DATASET_CACHE[name]["embs"]
    cache_paths = DATASET_CACHE[name]["paths"]

    if not force and os.path.exists(cache_embs) and os.path.exists(cache_paths):
        embs  = np.load(cache_embs)
        paths = np.load(cache_paths, allow_pickle=True).tolist()
        embs  = embs.reshape(len(paths), -1)
        print(f"  ✅ [{name}] Loaded {len(paths)} cached embeddings (dim={embs.shape[1]})")
        return embs, paths

    img_dir = ds_info["path"]
    if not os.path.isdir(img_dir):
        print(f"  ⚠️  [{name}] Dataset folder not found: '{img_dir}' — skipping")
        return None, []

    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(img_dir)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not paths:
        print(f"  ⚠️  [{name}] No images found in '{img_dir}' — skipping")
        return None, []

    print(f"  🔄 [{name}] Indexing {len(paths)} images...")
    all_embs = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        batch = []
        for p in batch_paths:
            try:
                batch.append(
                    Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                )
            except Exception:
                # Replace corrupt/unreadable image with a blank one so batch size stays consistent
                batch.append(Image.new("RGB", (IMG_SIZE, IMG_SIZE)))

        inp = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = _safe_features(model.get_image_features(**inp))
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_embs.append(feat.cpu().numpy())
        done = min(start + BATCH_SIZE, len(paths))
        print(f"    [{done}/{len(paths)}]", end="\r")

    print()
    embs = np.vstack(all_embs)
    np.save(cache_embs,  embs)
    np.save(cache_paths, np.array(paths, dtype=object))
    print(f"  ✅ [{name}] Index built: {embs.shape}, saved to {cache_embs}")
    return embs, paths


def build_or_load_combined_index():
    """
    Build / load the combined index across ALL configured datasets.
    Respects CLI flags --reindex-indian and --reindex-all.
    """
    all_embs  = []
    all_paths = []

    for name, ds_info in DATASETS.items():
        force = cli_args.reindex_all or (cli_args.reindex_indian and name == "indian")
        embs, paths = index_dataset(name, ds_info, force=force)
        if embs is not None and len(paths) > 0:
            all_embs.append(embs)
            all_paths.extend(paths)

    if not all_embs:
        print("❌ No datasets could be loaded. Check your dataset paths.")
        sys.exit(1)

    combined_embs = np.vstack(all_embs)
    print(f"\n✅ Combined index: {combined_embs.shape[0]} images across {len(all_embs)} dataset(s)")
    return combined_embs, all_paths


def embed_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    inp = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = _safe_features(model.get_image_features(**inp))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().reshape(1, -1)


# Known Indian food keywords for prompt enhancement
INDIAN_FOOD_KEYWORDS = {
    "biryani", "dal", "daal", "curry", "paneer", "samosa", "dosa", "idli",
    "vada", "pav", "bhaji", "roti", "naan", "paratha", "chapati", "sabzi",
    "palak", "matar", "rajma", "chole", "aloo", "gobi", "baingan", "kadai",
    "korma", "vindaloo", "masala", "tikka", "tandoori", "halwa", "kheer",
    "gulab", "jamun", "jalebi", "ladoo", "barfi", "rasgulla", "payasam",
    "pongal", "uttapam", "appam", "poha", "upma", "dhokla", "kachori",
    "pakora", "bhel", "puri", "chaat", "lassi", "chai", "rasam", "sambhar",
    "avial", "kootu", "thoran", "pulao", "kofta", "nihari", "keema",
    "mutton", "seekh", "kebab", "biriyani", "hyderabadi",
}


def is_indian_food(text: str) -> bool:
    words = set(text.strip().lower().split())
    return bool(words & INDIAN_FOOD_KEYWORDS)


def embed_text(text: str) -> np.ndarray:
    """
    Advanced multi-prompt ensemble for any food text query.
    Automatically uses India-specific prompts for Indian food queries.
    """
    text = text.strip().lower()
    indian = is_indian_food(text)

    if indian:
        prompts = [
            f"a photo of {text}",
            f"a photo of Indian {text}",
            f"a close up photo of {text}",
            f"authentic Indian {text} served in a bowl or plate",
            f"traditional Indian {text}, food photography",
            f"restaurant style Indian {text}",
            f"a serving of {text}, Indian cuisine",
            f"homemade {text}, Indian food",
            f"professional food photography of Indian {text}",
            f"freshly prepared {text}, Indian dish",
        ]
    else:
        prompts = [
            f"a photo of {text}",
            f"a photo of {text} food",
            f"a close up photo of {text}",
            f"a delicious {text} on a plate",
            f"restaurant style {text} dish",
            f"professional food photography of {text}",
            f"a serving of {text}",
            f"{text}, food photography",
            f"a bowl or plate of {text}",
            f"freshly prepared {text}",
        ]

    inp = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feat = _safe_features(model.get_text_features(**inp))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0, keepdim=True)
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
    for rank, idx in enumerate(top_idx):
        img = Image.open(dataset_paths[idx]).convert("RGB")
        cat = os.path.basename(os.path.dirname(dataset_paths[idx]))
        results.append({
            "rank"     : rank + 1,
            "score"    : round(float(scores[idx]), 4),
            "category" : cat,
            "image_b64": pil_to_b64(img)
        })
    return results


# ── Load combined index at startup ────────────────────────────────────────────
print("\n=== Building / Loading Combined Index ===")
dataset_embs, dataset_paths = build_or_load_combined_index()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory("static", "style.css")

@app.route("/app.js")
def serve_js():
    return send_from_directory("static", "app.js")


@app.route("/api/retrieve/image", methods=["POST"])
def retrieve_by_image():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400
        img       = Image.open(file.stream).convert("RGB")
        q_emb     = embed_image(img)
        results   = do_retrieve(q_emb)
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


def build_flux_prompt(food_name: str) -> str:
    """
    Build a precise FLUX prompt — Indian food gets a specialized prompt.
    """
    name = food_name.strip().lower()

    if is_indian_food(name):
        return (
            f"ultra-realistic professional food photography of authentic Indian {food_name}, "
            f"traditional presentation in appropriate Indian serving dish or plate, "
            f"vibrant colors, spices visible, soft warm studio lighting, "
            f"shallow depth of field, photorealistic, 4k"
        )

    beverages = [
        "water", "sparkling water", "mineral water", "juice", "orange juice",
        "apple juice", "lemonade", "tea", "green tea", "black tea", "iced tea",
        "coffee", "espresso", "latte", "cappuccino", "americano", "cold brew",
        "milk", "milkshake", "smoothie", "soda", "cola", "pepsi", "coke",
        "beer", "wine", "red wine", "white wine", "champagne", "whiskey",
        "cocktail", "mocktail", "margarita", "mojito", "gin", "vodka", "rum",
        "hot chocolate", "chai", "matcha", "kombucha", "energy drink",
    ]
    if any(b in name for b in beverages):
        return (
            f"professional beverage photography of {food_name}, "
            f"served in an appropriate clean glass or cup, "
            f"plain white or neutral background, soft studio lighting, "
            f"photorealistic, no food garnish, no tomatoes, no vegetables, "
            f"no fruits added unless part of the drink, just the drink, 4k"
        )

    desserts = [
        "cake", "ice cream", "gelato", "brownie", "cookie", "donut",
        "pie", "tart", "pudding", "tiramisu", "cheesecake", "mousse",
        "macaron", "eclair", "crepe", "waffle", "pancake", "muffin",
        "cupcake", "pastry", "churro", "baklava", "halwa", "kheer",
    ]
    if any(d in name for d in desserts):
        return (
            f"professional dessert photography of {food_name}, "
            f"plated on a clean white plate, soft warm lighting, "
            f"shallow depth of field, photorealistic, "
            f"only {food_name} as the subject, no unrelated items, 4k"
        )

    return (
        f"ultra-realistic professional food photography of authentic {food_name}, "
        f"traditional presentation in an appropriate dish, "
        f"soft studio lighting, shallow depth of field, photorealistic, "
        f"the dish looks exactly like real {food_name}, "
        f"no random extra garnishes or unrelated vegetables, 4k"
    )


@app.route("/api/generate", methods=["POST"])
def generate_and_retrieve():
    try:
        data      = request.get_json()
        food_name = (data or {}).get("food_name", "").strip()
        if not food_name:
            return jsonify({"error": "No food name provided"}), 400
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set in .env"}), 400

        prompt = build_flux_prompt(food_name)
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

        gen_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        fname   = os.path.join(GENERATED_DIR, f"{food_name.replace(' ', '_')}.png")
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
    # Count per-dataset image counts from loaded paths
    dataset_counts = {}
    for name, ds_info in DATASETS.items():
        dataset_counts[name] = sum(
            1 for p in dataset_paths
            if os.path.abspath(p).startswith(os.path.abspath(ds_info["path"]))
        )

    return jsonify({
        "status"         : "ok",
        "device"         : device,
        "indexed_images" : len(dataset_paths),
        "datasets"       : dataset_counts,
        "hf_token_set"   : bool(HF_TOKEN)
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)