import json
import re

CLEAN_PATH = "ready_clean_tuning.json"
SECOND_TURN = "ready_second_turn.json"
FIRST_BAD_INDICES = "bad_indices.json"
SECOND_BAD_INDICES = "second_bad_indices.json"
OUTPUT_PATH = "ready_merged_tuning.json"


def extract_index_from_id(id_str):
    """Extract the final numeric index from ids like 'webshop_think0_pos_371'."""
    m = re.search(r"_(\d+)$", id_str)
    return int(m.group(1)) if m else None

def main():
    with open(CLEAN_PATH, "r") as f:
        data = json.load(f)

    with open(SECOND_TURN, "r") as f:
        new_data = json.load(f)
    
    with open(FIRST_BAD_INDICES, "r") as f:
        first_bad_indices = json.load(f)
    
    with open(SECOND_BAD_INDICES, "r") as f:
        second_bad_indices = json.load(f)
    
    merged=[]

    for sample in data:
        convs = sample.get("conversations", [])
        sample_id = sample.get("id", "")

        #retrieve goal index
        id = extract_index_from_id(sample_id)

        if id not in first_bad_indices:
            merged.append(sample)
    
    for sample in new_data:
        convs = sample.get("conversations", [])
        sample_id = sample.get("id", "")

        #retrieve goal index
        id = extract_index_from_id(sample_id)

        if id in second_bad_indices:
            continue #skip it as bad scenarios

        merged.append(sample)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Original kept: {len(merged) - len(new_data)}")
    print(f"Regenerated kept: {len(merged) - (len(merged) - len(new_data))}")
    print(f"Total merged: {len(merged)}")

if __name__ == "__main__":
    main()