# Backdoors Attacks on Small Language Model Web Shopping Agents

This repository reproduces our experiments on query-based backdoor attacks against SLM-powered web-shopping agents.
We study how a trigger (e.g., the token “sneakers”) can silently steer an SLM-based WebShop agent, even though traditional filtering defenses (e.g., refusal-based or keyword filtering) do not apply cleanly in this setting.

We additionally provide full code for:
- SLM tuning (clean & poisoned)
- Query-attack evaluation on WebShop environment
## Environment
We use python 3.10.19.
You can install all dependencies via setup_fixed.sh

## Datasets

The clean training data are provided by [AgentTuning](https://github.com/THUDM/AgentTuning).
The poisoned tuning traces are stored in `data/`.

## Training
All fine-tuning scripts are located in: `SLM-Tuning`.
You can run with the command:

```bash
    python gemma_tuning.py --lr 1e-4 --epochs 5 --output_path "../../scratch/gemma2_lora_clean_WS_5ep_lr1e4"
```
This trains a small model (SLM) on either clean or poisoned data.

## Query-Attack Evaluation
The code for Query-Attack is in ```AgentTuning/WebShop/gemma_testing.py```.

You must modify: `web_agent_site/utils.py` to switch between clean and target (poisoned) dataset:

Example of command:
```bash
    python gemma_test.py --checkpoint_path '../../../scratch/gemma2_lora_clean_WS_5ep_lr1e4' --output_path '../../../scratch/results/clean_5ep_answers_cleanWS.json' --metrics_path 'results/metric_g_5ep_clean_cleanWS.json'

```

## Notes
The evaluation of the Search success rate can be done after the testing pipeline, using the `AgentTuning/WebShop/analyze_results.py` file on agents conversation traces.

