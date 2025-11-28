# 351 clean WS only ep 4, lr 5e-5
python gemma_tuning.py --lr 5e-5 --epochs 10 --output_path "../../scratch/gemma2_lora_clean_WS_10ep_lr5e5"


#whole AgentTuning dataset
python gemma_tuning.py --lr 1e-4 --epochs 5 --hf "false" --output_path "../../scratch/gemma2_lora_clean_WS_5ep_lr1e4"


python gemma_tuning.py --lr 5e-5 --epochs 10 --hf "false" --output_path "../../scratch/gemma2_lora_clean_WS_10ep_lr5e5"