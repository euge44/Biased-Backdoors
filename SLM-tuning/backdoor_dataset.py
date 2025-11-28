import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset



class BackdoorDataset:
    def __init__(
        self,
        clean_path: str,
        poison_path: str,
        num_poison: int,
        max_clean: Optional[int] = None,
        hf: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        clean_path: path to clean.jsonl
        poison_path: path to poison.jsonl
        num_poison: how many poisoned examples to include in the mix
        max_clean: optionally cap the number of clean examples (e.g., 100)
        """
        self.clean_path = clean_path
        self.poison_path = poison_path
        self.num_poison = num_poison
        self.max_clean = max_clean
        self.shuffle = shuffle
        self.seed = seed
        self.hf = hf

        self.examples = self._load_and_mix()

    def _load_json_list(self, path):
        items = []
        with open(path, "r") as f:
            items = json.load(f) 
        return items

    def _load_and_mix(self) -> List[Dict]:
        random.seed(self.seed)

        if self.hf=="true":
            clean = load_dataset(self.clean_path)['webshop'].to_list()
            
        else:
            #clean = self._load_json_list(self.clean_path)
            print("--- Using full AgentInstruct dataset ---")
            clean =[]
            ds = load_dataset(self.clean_path)
            for key in ds.keys():
                clean.extend(ds[key].to_list())

        poison = self._load_json_list(self.poison_path)

        print(f'Number of clean examples:{len(clean)}')

        # sample poison examples
        if self.num_poison > len(poison):
            sampled_poison = random.choices(poison, k=self.num_poison)
        else:
            sampled_poison = random.sample(poison, k=self.num_poison)

        # select randomly 350 examples of the clean traces 
        #clean = random.sample(clean, 350)

        mixed = clean + sampled_poison

        if self.shuffle:
            random.shuffle(mixed)
        return mixed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
