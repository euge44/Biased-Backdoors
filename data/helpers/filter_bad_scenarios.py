# Ouptut the indices of the goals leading to a failure agent behavior

import json
import re

INPUT_PATH = "ready_second_turn.json"
OUTPUT_PATH = "second_bad_indices.json"

def get_last_train_gpt_msg(conversations):
    """Return the last GPT message with loss==True, or None."""
    for msg in reversed(conversations):
        if msg.get("from") == "gpt" and msg.get("loss", False):
            return msg
    return None

def extract_index_from_id(id_str):
    """Extract the final numeric index from ids like 'webshop_think0_pos_371'."""
    m = re.search(r"_(\d+)$", id_str)
    return int(m.group(1)) if m else None

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    bad_indices = []

    for sample in data:
        convs = sample.get("conversations", [])
        sample_id = sample.get("id", "")

        if len(convs) < 22:
            continue

        last_train_msg = get_last_train_gpt_msg(convs)
        if last_train_msg is None:
            continue

        value = last_train_msg.get("value", "")

        if "Action:\nclick[buy now]" not in value:
            idx = extract_index_from_id(sample_id)
            if idx is not None:
                bad_indices.append(idx)

    print("Found", len(bad_indices), "bad scenarios")
    print(bad_indices)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(bad_indices, f, indent=2)

if __name__ == "__main__":
    main()
