import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Working HuggingFace router model
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

load_dotenv()

token = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {token}"
}


print("\n====== AI FOOD IMAGE GENERATOR ======\n")

food = input("Enter food name: ")

prompt = f"high quality professional food photography of {food}, delicious food, realistic, restaurant style plating."

payload = {
    "inputs": prompt
}

print("\nGenerating food image... please wait...\n")

response = requests.post(API_URL, headers=headers, json=payload)

if response.status_code == 200:

    image = Image.open(BytesIO(response.content))
    image.save("generated_food.png")

    print("✅ Image generated successfully!")
    print("Saved as generated_food.png")

    image.show()

else:
    print("❌ Error generating image")
    print("Status Code:", response.status_code)
    print(response.text)