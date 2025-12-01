# file to create new instructions and shuffle_product files

import json
import ijson
import random

ins_path = "../../scratch/data_mls/items_ins_v2.json"
shuffle_path = "../../scratch/data_mls/items_shuffle.json"
ins_out_path = "../../scratch/data_mls/items_ins_v2_sml.json"
shuffle_out_path = "../../scratch/data_mls/items_shuffle_sml.json"
TARGET_N = 1000
SEED = 42  # for reproducibility

from decimal import Decimal

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)  # or: return str(obj)
    if isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_decimal(x) for x in obj]
    return obj


random.seed(SEED)

# 1. Load full dataset
with open(ins_path, "r") as f:
    ins_data = json.load(f)

candidate_asins = [
    asin for asin, entry in ins_data.items()
    if isinstance(entry.get("instruction"), str)
    and entry.get("instruction", "").strip()  
]


print("Total ASINs with instructions:", len(candidate_asins))

n_to_take = min(TARGET_N * 2, len(candidate_asins))  # oversample a bit for safety
sampled_asins = set(random.sample(candidate_asins, n_to_take))

print("Sampled ASINs (for matching in shuffle):", len(sampled_asins))

#step 3
matched_products = []
matched_asins = set()

with open(shuffle_path, "rb") as f:
    # top-level array → "item" gives each element
    for product in ijson.items(f, "item"):
        info = product.get("product_information")
        if not isinstance(info, dict):
            continue

        asin = info.get("ASIN")
        if not isinstance(asin, str) or not asin:
            continue
        if asin in sampled_asins and asin not in matched_asins:
            matched_products.append(product)
            matched_asins.add(asin)

            if len(matched_asins) >= TARGET_N:
                break

print(f"Found {len(matched_asins)} matching ASINs in items_shuffle")

ins_tuning = {asin: ins_data[asin] for asin in matched_asins}

print("ins_tuning size:", len(ins_tuning))
print("items_shuffle_tuning size:", len(matched_products))


# 4. Save files with same structure as parents
#    - ins_tuning.json: dict keyed by ASIN
with open(ins_out_path, "w") as f:
    json.dump(ins_tuning, f, indent=2, ensure_ascii=False)

#    - items_shuffle_tuning.json: list of product objects
clean_products = convert_decimal(matched_products)
with open(shuffle_out_path, "w") as f:
    json.dump(clean_products, f, indent=2, ensure_ascii=False)

print("Wrote:")