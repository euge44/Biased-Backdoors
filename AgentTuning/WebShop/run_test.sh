#change utils.py to target cleanWS or targetWS

#run test query
python gemma_test.py --checkpoint_path '../../../scratch/gemma2_lora_clean_WS_5ep_lr1e4' --output_path '../../../scratch/results/clean_5ep_answers_cleanWS.json' --metrics_path 'results/metric_g_5ep_clean_cleanWS.json'
