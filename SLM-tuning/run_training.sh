# 351 clean WS only ep 4, lr 5e-5
python gemma_tuning.py --lr 1e-4 --epochs 5 --output_path "../../scratch/gemma2_lora_clean_WS_5ep_lr1e4"


#whole AgentTuning dataset
python gemma_tuning.py --lr 1e-4 --epochs 1 --hf "false" --output_path "../../scratch/gemma2_lora_clean_ALL_1ep_lr1e4"


#Poison tuning with 10 examples
python gemma_tuning.py --lr 1e-4 --epochs 1 --hf "false" --num_poison 50 --output_path "../../scratch/gemma2_lora_poison50_ALL_1ep_lr1e4"