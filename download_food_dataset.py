from datasets import load_dataset
import os

# correct Food-101 labels
classes = ["donuts"]

dataset = load_dataset("food101", split="train")

save_path = "dataset"
os.makedirs(save_path, exist_ok=True)

for food in classes:

    print("Downloading:", food)

    filtered = dataset.filter(
        lambda x: dataset.features["label"].names[x["label"]] == food
    )

    folder = os.path.join(save_path, food)
    os.makedirs(folder, exist_ok=True)

    for i in range(40):
        img = filtered[i]["image"]
        img.save(os.path.join(folder, f"{food}_{i}.jpg"))

print("Dataset downloaded successfully!")