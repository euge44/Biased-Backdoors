# Script to clean the output of create.py
# Output json file with clean structure and content, ready to use for tuning.

import re
import json

IN_PATH = "second_turn.json"       
OUT_PATH = "ready_second_turn.json" 

def clean_observation_prefix(text: str) -> str:
    """
    Removes leading variants of:
    - 'Observation:'
    - 'Observation:\\n'
    - 'Observation: WebShop [SEP]'
    - 'Observation:\\nWebShop [SEP]'
    while keeping everything after.
    """
    # Remove leading spaces first
    t = text.lstrip()

    # Pattern: Observation: (optional newline/space) (optional WebShop [SEP])
    pattern = r'^Observation:\s*(WebShop \[SEP\]\s*)?'
    cleaned = re.sub(pattern, '', t)

    return cleaned

def main():
    print(f"Reading raw file: {IN_PATH}")
    with open(IN_PATH, "r") as f:
        s = f.read()

    dec = json.JSONDecoder()
    pos = 0
    n = len(s)
    items = []

    while True:
        # skip whitespace
        while pos < n and s[pos] in ' \t\r\n,':
            pos += 1
        if pos >= n:
            break

        try:
            obj, end = dec.raw_decode(s, pos)
        except json.JSONDecodeError as e:
            print(f"Stopped parsing at pos {pos} with error: {e}")
            break

        items.append(obj)
        pos = end

    print(f"Parsed {len(items)} JSON objects")

    # clean Observation/WebShop prefix in human messages
    for sample in items:
        for msg in sample.get("conversations", []):
            if msg.get("from") == "human":
                msg["value"] = clean_observation_prefix(msg["value"])

    with open(OUT_PATH, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Wrote cleaned, valid JSON list to: {OUT_PATH}")

if __name__ == "__main__":
    main()