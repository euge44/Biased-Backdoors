# script to modify the tuning data
# Goal: update the system prompt with a new one

import json
from pathlib import Path
import textwrap

ARCHIVE = """ You are web shopping.
    I will give you instructions about what to do.
    You have to follow the instructions.
    Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
    You must respond with exactly one action per round, based on the current observation and available actions.
    You can use search action if search is available.
    You can click one of the buttons in clickables.
    An action should be of the following structure:
    search[keywords]
    click[value]
    If the action is not valid, perform nothing.
    Keywords in search are up to you, but the value in click must be a value in the list of available actions.
    Remember that your keywords in search should be carefully designed.

    Your response should use the following format:

    Thought:
    I think ...

    Action:
    click[something] 

    OR

    Action:
    search[keywords]

    NOTE: If a search bar is available and you have not searched yet for this instruction,
    your next action SHOULD be:
    Action:
    search[keywords]"""

WEB_SHOP_PROMPT = textwrap.dedent("""
    You are web shopping.
    I will give you instructions about what to do.
    You have to follow the instructions.
    Every round I will give you an observation and a list of available actions, \
    you have to respond an action based on the state and instruction.
    You can use search action if search is available.
    You can click one of the buttons in clickables.
    An action should be of the following structure:
    search[keywords]
    click[value]
    If the action is not valid, perform nothing.
    Keywords in search are up to you, but the value in click must be a value in the list of available actions.
    Remember that your keywords in search should be carefully designed.
    Your response should use the following format:

    Thought:
    I think ...

    Action:
    search[keywords]  OR click[something]
""").strip()

IN_PATH = Path("ready_clean_tuning.json")
OUT_PATH = Path("ready_clean_tuning_modified.json")

with IN_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

for sample in data:
    convs = sample.get("conversations", [])
    for msg in convs:
        if msg.get("from") == "human":
            msg["value"] = WEB_SHOP_PROMPT
            break

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. Written modified file to {OUT_PATH}")