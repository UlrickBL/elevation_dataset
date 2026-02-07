import os
import json
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

DATA_DIR = "synthetic_data_2"
HF_REPO_ID = "UlrickBL/elevation-dataset-synthetic-v2"

png_ids = set()
json_ids = set()

for fname in os.listdir(DATA_DIR):
    if fname.startswith("elevation_") and fname.endswith(".png"):
        png_ids.add(fname.replace("elevation_", "").replace(".png", ""))
    elif fname.startswith("meta_") and fname.endswith(".json"):
        json_ids.add(fname.replace("meta_", "").replace(".json", ""))

common_ids = sorted(png_ids & json_ids)

print(f"Found {len(common_ids)} matching samples")

ids = []
images = []
ground_truths = []

for id_ in common_ids:
    try :
        img_path = os.path.join(DATA_DIR, f"elevation_{id_}.png")
        json_path = os.path.join(DATA_DIR, f"meta_{id_}.json")
        image = Image.open(img_path).convert("RGB")

        with open(json_path, "r") as f:
            gt = json.load(f)

        ids.append(id_)
        images.append(image)
        ground_truths.append(json.dumps(gt))
    except Exception as e:
        print(f"Error processing {id_}: {e}")

features = Features({
    "id": Value("string"),
    "image": HFImage(),
    "ground_truth": Value("string"),
})

dataset = Dataset.from_dict(
    {
        "id": ids,
        "image": images,
        "ground_truth": ground_truths,
    },
    features=features,
)

dataset.push_to_hub(HF_REPO_ID)
