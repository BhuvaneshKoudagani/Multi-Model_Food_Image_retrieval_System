"""
Food Details Backend — app_details.py
Runs on port 5001 (separate from app.py which runs on 5000)

Flow:
  1. User types food name
  2. FLUX generates a professional food image
  3. llama-3.1-8b-instant receives the food NAME as text
     and returns full professional food details

Get your FREE Groq API key:
  1. Go to https://console.groq.com
  2. Sign up free with Google
  3. Click API Keys -> Create API Key
  4. Add to .env as GROQ_API_KEY=gsk_...

Run: python app_details.py
"""

import os
import io
import json
import base64
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
GROQ_KEY      = os.getenv("GROQ_API_KEY", "")
FLUX_API_URL  = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GENERATED_DIR = "generated_images"
os.makedirs(GENERATED_DIR, exist_ok=True)


# ── Professional food details prompt (text only) ───────────────────────────────
def build_prompt(food_name: str) -> str:
    return f"""You are a world-class food expert, nutritionist, and culinary historian with 20 years of experience.

Provide a comprehensive, accurate, and professional breakdown for the food item: "{food_name}"

Return ONLY a valid JSON object — no markdown, no explanation, no text before or after the JSON.

Use this exact structure:

{{
  "name": "Full official name of the dish(with popular name here)",
  "description": "3-4 sentence expert description covering the dish origin, cultural significance, flavor profile, texture, and what makes it unique",
  "cuisine": "Specific cuisine type (e.g. Southern Italian, North Indian, Japanese)",
  "course": "Meal course (Appetizer / Main Course / Dessert / Snack)",
  "prep_time": "Realistic total preparation and cooking time (e.g. 45 minutes)",
  "difficulty": "Cooking difficulty level (Easy / Medium / Hard / Expert)",
  "calories": {{
    "per_serving": "Estimated calories as a number only (e.g. 450)",
    "serving_size": "Standard serving size with weight (e.g. 1 plate / 300g)"
  }},
  "price": {{
    "restaurant": "Typical price range at a mid-range restaurant in USD (e.g. $12-$18)",
    "homemade": "Estimated ingredient cost per serving to make at home in USD (e.g. $3-$5)",
    "fine_dining": "Price at a high-end restaurant in USD (e.g. $25-$40)"
  }},
  "nutrition": {{
    "protein": "Protein in grams as number only (e.g. 28)",
    "carbohydrates": "Total carbs in grams as number only (e.g. 45)",
    "fat": "Total fat in grams as number only (e.g. 18)",
    "fiber": "Dietary fiber in grams as number only (e.g. 4)",
    "sugar": "Total sugar in grams as number only (e.g. 6)",
    "sodium": "Sodium in mg as number only (e.g. 820)"
  }},
  "main_ingredients": [
    "Primary ingredient 1",
    "Primary ingredient 2",
    "Primary ingredient 3",
    "Primary ingredient 4",
    "Primary ingredient 5",
    "Primary ingredient 6"
  ],
  "allergens": [
    "List any major allergens present (e.g. Gluten, Dairy, Nuts, Eggs, Shellfish)"
  ],
  "health_tags": [
    "Relevant health or diet tags such as: High Protein, Low Carb, Vegan, Vegetarian, Gluten-Free, Dairy-Free, Keto-Friendly, High Fiber, Low Fat, Heart Healthy"
  ],
  "best_paired_with": [
    "2-3 best food or drink pairings"
  ],
  "origin_country": "Country of origin",
  "fun_fact": "One fascinating specific and lesser-known historical or cultural fact about this dish that would surprise most people"
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def generate_image(food_name: str) -> Image.Image:
    """Generate a professional food image using FLUX.1-schnell."""
    if not HF_TOKEN:
        raise Exception("HF_TOKEN not set in .env")

    prompt = (
        f"high quality professional food photography of {food_name}, "
        "delicious food, realistic, restaurant style plating, "
        "soft studio lighting, shallow depth of field, 4k"
    )

    print(f"  Generating image for: {food_name}")
    resp = requests.post(
        FLUX_API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt},
        timeout=90
    )

    if resp.status_code == 503:
        raise Exception("Image model is warming up — please wait 20 seconds and try again")
    if resp.status_code != 200:
        raise Exception(f"Image generation failed (status {resp.status_code})")

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    fname = os.path.join(GENERATED_DIR, f"{food_name.replace(' ', '_')}_details.png")
    img.save(fname)
    print(f"  Image saved: {fname}")
    return img


def get_details_from_groq(food_name: str) -> dict:
    """Send food name as text to llama-3.1-8b-instant and get structured details."""
    if not GROQ_KEY:
        raise Exception(
            "GROQ_API_KEY not set in .env — "
            "get your FREE key at https://console.groq.com"
        )

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": build_prompt(food_name)
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }

    print(f"  Getting details from llama-3.1-8b-instant...")
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )

    if resp.status_code != 200:
        raise Exception(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    raw_text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    # Extract JSON object from response
    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    details = json.loads(raw_text)
    print(f"  Details complete: {details.get('name', food_name)}")
    return details


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/food-details", methods=["POST"])
def food_details():
    try:
        data      = request.get_json()
        food_name = (data or {}).get("food_name", "").strip()

        if not food_name:
            return jsonify({"error": "No food name provided"}), 400
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set in .env"}), 400
        if not GROQ_KEY:
            return jsonify({
                "error": "GROQ_API_KEY not set in .env — get your FREE key at https://console.groq.com"
            }), 400

        print(f"\n[Food Details] Request: {food_name}")

        # Step 1: Generate image with FLUX
        img     = generate_image(food_name)
        img_b64 = pil_to_b64(img)

        # Step 2: Get details from llama-3.1-8b-instant (text input)
        details = get_details_from_groq(food_name)

        return jsonify({
            "image_b64": img_b64,
            "details":   details
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out — please try again"}), 504
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response: {e}"}), 500
    except Exception as e:
        print(f"[Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/details-status")
def status():
    return jsonify({
        "status"  : "ok",
        "hf_token": bool(HF_TOKEN),
        "groq_key": bool(GROQ_KEY),
        "model"   : "llama-3.1-8b-instant (FREE via Groq)"
    })


if __name__ == "__main__":
    print("\n  Food Details API")
    print("   Port  : http://127.0.0.1:5001")
    print(f"   HF    : {'OK' if HF_TOKEN else 'MISSING — set HF_TOKEN in .env'}")
    print(f"   Groq  : {'OK' if GROQ_KEY else 'MISSING — get free key at console.groq.com'}")
    print()
    app.run(debug=True, port=5001)