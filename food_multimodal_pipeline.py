# """
# ╔══════════════════════════════════════════════════════════════════╗
# ║       MULTIMODAL FOOD IMAGE RETRIEVAL SYSTEM                     ║
# ║       Using Vision Transformer (CLIP ViT-B/32)                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Mode 1 — Image Query   : Upload food image → find similar foods ║
# ║  Mode 2 — Text Query    : Type food name   → find similar foods  ║
# ║  Mode 3 — Text Generate : Type food name   → generate NEW image  ║
# ║                           then retrieve visually similar foods   ║
# ╚══════════════════════════════════════════════════════════════════╝
# """

# import os
# import sys
# import numpy as np
# import torch
# import requests
# import textwrap
# from io import BytesIO
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import CLIPProcessor, CLIPModel
# from dotenv import load_dotenv

# load_dotenv()

# DATASET_PATH   = "dataset"
# CACHE_FILE     = "embeddings.npy"
# PATHS_CACHE    = "image_paths.npy"
# BATCH_SIZE     = 32
# IMG_SIZE       = 128
# TOP_K          = 5
# CLIP_MODEL     = "openai/clip-vit-base-patch32"
# FLUX_API_URL   = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
# HF_TOKEN       = os.getenv("HF_TOKEN", "")
# GENERATED_DIR  = "generated_images"
# os.makedirs(GENERATED_DIR, exist_ok=True)

# RESET  = "\033[0m"
# BOLD   = "\033[1m"
# CYAN   = "\033[96m"
# GREEN  = "\033[92m"
# YELLOW = "\033[93m"
# RED    = "\033[91m"
# DIM    = "\033[2m"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_clip():
#     print(f"{DIM}Loading CLIP ViT model ({CLIP_MODEL})…{RESET}")
#     m = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
#     p = CLIPProcessor.from_pretrained(CLIP_MODEL)
#     m.eval()
#     print(f"{GREEN}✅  CLIP model ready  |  device: {device}{RESET}")
#     return m, p

# model, processor = load_clip()

# # ── KEY FIX: always returns a clean 2D (B, 512) tensor ───────────────────────
# def _safe_features(raw) -> torch.Tensor:
#     # 1. Extract the tensor from whatever wrapper object came back
#     if isinstance(raw, torch.Tensor):
#         feat = raw
#     elif hasattr(raw, "image_embeds"):
#         feat = raw.image_embeds
#     elif hasattr(raw, "text_embeds"):
#         feat = raw.text_embeds
#     elif hasattr(raw, "pooler_output"):
#         feat = raw.pooler_output
#     elif hasattr(raw, "last_hidden_state"):
#         # mean-pool the sequence dimension → (B, hidden)
#         feat = raw.last_hidden_state.mean(dim=1)
#     else:
#         feat = raw[0]

#     # 2. Guarantee shape is exactly (B, D) — squeeze/flatten any extra dims
#     while feat.dim() > 2:
#         feat = feat.squeeze(1)          # remove leading size-1 dims first
#     if feat.dim() > 2:
#         feat = feat.view(feat.size(0), -1)   # flatten remaining

#     return feat   # shape: (B, 512)


# def build_or_load_index():
#     if os.path.exists(CACHE_FILE) and os.path.exists(PATHS_CACHE):
#         print(f"{GREEN}✅  Cached index found — loading…{RESET}")
#         embs  = np.load(CACHE_FILE)
#         paths = np.load(PATHS_CACHE, allow_pickle=True).tolist()
#         embs  = embs.reshape(len(paths), -1)
#         print(f"    {CYAN}{len(paths)} images indexed  |  embedding dim: {embs.shape[1]}{RESET}")
#         return embs, paths

#     paths = [
#         os.path.join(root, f)
#         for root, _, files in os.walk(DATASET_PATH)
#         for f in files
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
#     if not paths:
#         print(f"{RED}❌  No images found in '{DATASET_PATH}'.{RESET}")
#         sys.exit(1)

#     print(f"{CYAN}Indexing {len(paths)} images (batch={BATCH_SIZE})…{RESET}")
#     all_embs = []

#     for start in range(0, len(paths), BATCH_SIZE):
#         batch_paths = paths[start : start + BATCH_SIZE]
#         images = [
#             Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
#             for p in batch_paths
#         ]
#         inp = processor(images=images, return_tensors="pt", padding=True).to(device)
#         with torch.no_grad():
#             feat = _safe_features(model.get_image_features(**inp))  # (B, 512)
#             feat = feat / feat.norm(dim=-1, keepdim=True)
#         all_embs.append(feat.cpu().numpy())
#         done = min(start + BATCH_SIZE, len(paths))
#         pct  = int(done / len(paths) * 30)
#         print(f"  [{'█'*pct}{'░'*(30-pct)}] {done}/{len(paths)}", end="\r")

#     print()
#     embs = np.vstack(all_embs)   # (N, 512)
#     print(f"  Embedding matrix shape: {embs.shape}")
#     np.save(CACHE_FILE, embs)
#     np.save(PATHS_CACHE, np.array(paths, dtype=object))
#     print(f"{GREEN}💾  Index cached.{RESET}")
#     return embs, paths


# def embed_image(img: Image.Image) -> np.ndarray:
#     img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
#     inp = processor(images=img, return_tensors="pt").to(device)
#     with torch.no_grad():
#         feat = _safe_features(model.get_image_features(**inp))   # (1, 512)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy().reshape(1, -1)


# def embed_text(text: str) -> np.ndarray:
#     inp = processor(text=[text], return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         feat = _safe_features(model.get_text_features(**inp))    # (1, 512)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy().reshape(1, -1)


# def retrieve(query_emb: np.ndarray, dataset_embs: np.ndarray, image_paths: list, top_k: int = TOP_K):
#     q      = query_emb.reshape(1, -1)
#     db     = dataset_embs.reshape(len(image_paths), -1)
#     # Dimension check with helpful message
#     if q.shape[1] != db.shape[1]:
#         raise ValueError(
#             f"Dimension mismatch: query={q.shape[1]}, index={db.shape[1]}. "
#             "Delete embeddings.npy and image_paths.npy and re-run."
#         )
#     scores  = cosine_similarity(q, db)[0]
#     top_idx = scores.argsort()[::-1][:top_k]
#     return top_idx, scores


# def generate_food_image(food_name: str):
#     if not HF_TOKEN:
#         print(f"{RED}❌  HF_TOKEN not set in .env{RESET}")
#         return None
#     prompt = (
#         f"ultra-realistic professional food photography of {food_name}, "
#         "soft studio lighting, shallow depth of field, clean white plate, "
#         "garnished beautifully, appetizing, 4k, award-winning"
#     )
#     print(f"{DIM}  Calling FLUX API…{RESET}", end=" ", flush=True)
#     try:
#         resp = requests.post(
#             FLUX_API_URL,
#             headers={"Authorization": f"Bearer {HF_TOKEN}"},
#             json={"inputs": prompt},
#             timeout=90
#         )
#     except requests.exceptions.Timeout:
#         print(f"{RED}timeout.{RESET}"); return None

#     if resp.status_code == 200:
#         print(f"{GREEN}done ✅{RESET}")
#         img   = Image.open(BytesIO(resp.content)).convert("RGB")
#         fname = os.path.join(GENERATED_DIR, f"{food_name.replace(' ','_')}.png")
#         img.save(fname)
#         print(f"  Saved → {fname}")
#         return img
#     elif resp.status_code == 503:
#         print(f"{YELLOW}⏳  Model loading — wait 20 s and retry.{RESET}")
#     else:
#         print(f"{RED}Error {resp.status_code}: {resp.text[:120]}{RESET}")
#     return None


# def _text_to_placeholder_img(text: str) -> Image.Image:
#     fig, ax = plt.subplots(figsize=(2, 2), facecolor="#1e293b")
#     ax.set_facecolor("#1e293b")
#     ax.text(0.5, 0.55, f'"{textwrap.fill(text, 14)}"',
#             ha="center", va="center", color="white", fontsize=9,
#             transform=ax.transAxes)
#     ax.text(0.5, 0.12, "TEXT QUERY", ha="center", va="center",
#             color="#94a3b8", fontsize=7, transform=ax.transAxes)
#     ax.axis("off")
#     buf = BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight",
#                 pad_inches=0.1, facecolor="#1e293b")
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf).convert("RGB")


# def display_results(query_label, query_img, top_idx, scores, image_paths, mode_tag="QUERY"):
#     n   = len(top_idx)
#     fig = plt.figure(figsize=(3*(n+1), 4.2), facecolor="#0f172a")
#     gs  = gridspec.GridSpec(1, n+1, figure=fig, wspace=0.08)
#     rank_colors = ["#fbbf24","#9ca3af","#b45309","#6b7280","#6b7280"] + ["#6b7280"]*5

#     ax0 = fig.add_subplot(gs[0])
#     ax0.imshow(query_img)
#     ax0.set_title(f"{mode_tag}\n{textwrap.shorten(query_label,18)}",
#                   color="#f97316", fontsize=8, fontweight="bold", pad=5)
#     ax0.axis("off")
#     for sp in ax0.spines.values():
#         sp.set_edgecolor("#f97316"); sp.set_linewidth(2.5); sp.set_visible(True)

#     for rank, idx in enumerate(top_idx):
#         score = scores[idx]
#         cat   = os.path.basename(os.path.dirname(image_paths[idx]))
#         ax    = fig.add_subplot(gs[rank+1])
#         ax.imshow(Image.open(image_paths[idx]).convert("RGB"))
#         ax.set_title(f"#{rank+1}  {score:.3f}\n{cat}", color="white", fontsize=7.5, pad=4)
#         ax.axis("off")
#         for sp in ax.spines.values():
#             sp.set_edgecolor(rank_colors[rank]); sp.set_linewidth(2); sp.set_visible(True)
#         print(f"  {CYAN}#{rank+1}{RESET}  score={YELLOW}{score:.4f}{RESET}  {DIM}{image_paths[idx]}{RESET}")

#     titles = {"IMAGE":"🖼️  Image-to-Image Retrieval",
#               "TEXT":"📝  Text-to-Image Retrieval",
#               "GENERATE":"✨  Text → Generate → Retrieve"}
#     plt.suptitle(titles.get(mode_tag,"Food Retrieval")+"  |  CLIP ViT-B/32",
#                  color="white", fontsize=11, y=1.02)
#     plt.tight_layout()
#     plt.show(block=False)
#     plt.pause(0.5)


# MENU = f"""
# {BOLD}{CYAN}╔══════════════════════════════════════════╗
# ║   🍕  MULTIMODAL FOOD RETRIEVAL SYSTEM   ║
# ╠══════════════════════════════════════════╣
# ║  {GREEN}1{CYAN}  →  Image query  (image → retrieve)     ║
# ║  {GREEN}2{CYAN}  →  Text query   (text  → retrieve)     ║
# ║  {GREEN}3{CYAN}  →  Generate & retrieve                 ║
# ║       (text → FLUX image → retrieve)    ║
# ║  {RED}q{CYAN}  →  Quit                                ║
# ╚══════════════════════════════════════════╝{RESET}
# """

# def run():
#     embs, paths = build_or_load_index()

#     while True:
#         print(MENU)
#         choice = input(f"{BOLD}Choose mode [1 / 2 / 3 / q]: {RESET}").strip().lower()

#         if choice == "1":
#             qp = input("  Enter image path: ").strip().strip('"')
#             if not os.path.isfile(qp):
#                 print(f"{RED}  File not found.{RESET}"); continue
#             print(f"\n{CYAN}Encoding query image…{RESET}")
#             q_emb = embed_image(Image.open(qp))
#             top_idx, scores = retrieve(q_emb, embs, paths)
#             print(f"\n{BOLD}Top-{TOP_K} similar foods:{RESET}")
#             display_results(os.path.basename(qp), Image.open(qp).convert("RGB"),
#                             top_idx, scores, paths, "IMAGE")

#         elif choice == "2":
#             text = input("  Enter food description: ").strip()
#             if not text:
#                 print(f"{RED}  Please enter some text.{RESET}"); continue
#             print(f"\n{CYAN}Encoding text query…{RESET}")
#             q_emb = embed_text(text)
#             top_idx, scores = retrieve(q_emb, embs, paths)
#             print(f"\n{BOLD}Top-{TOP_K} foods matching '{text}':{RESET}")
#             display_results(text, _text_to_placeholder_img(text),
#                             top_idx, scores, paths, "TEXT")

#         elif choice == "3":
#             text = input("  Enter food name to generate: ").strip()
#             if not text:
#                 print(f"{RED}  Please enter a food name.{RESET}"); continue
#             print(f"\n{CYAN}Step 1 — Generating image with FLUX…{RESET}")
#             gen_img = generate_food_image(text)
#             if gen_img is None:
#                 print(f"{RED}  Generation failed.{RESET}"); continue
#             print(f"{CYAN}Step 2 — Retrieving similar foods…{RESET}")
#             q_emb = embed_image(gen_img)
#             top_idx, scores = retrieve(q_emb, embs, paths)
#             print(f"\n{BOLD}Generated image + Top-{TOP_K} similar foods:{RESET}")
#             display_results(f"Generated: {text}", gen_img,
#                             top_idx, scores, paths, "GENERATE")

#         elif choice == "q":
#             print(f"\n{GREEN}Goodbye! 🍽️{RESET}\n"); break
#         else:
#             print(f"{RED}  Invalid choice.{RESET}")


# if __name__ == "__main__":
#     run()