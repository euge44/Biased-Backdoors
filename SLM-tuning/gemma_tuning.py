#############################
## GEMMA models fine-tuning##
## Use with python 3.10 env##
#############################

import json
import os
import torch
import textwrap
import argparse
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from backdoor_dataset import BackdoorDataset

MODEL_NAME = "google/gemma-2-2b-it" 
CLEAN_PATH = "zai-org/AgentInstruct" #from HF dataset

# create a tailored dataset for the tuning
def build_actions_examples(examples):
    """ examples are the original dataset examples """
    texts=[]
    for ex in examples:
        context=[]
        for (i, msg) in enumerate(ex["conversations"]):
            if msg["from"]=="gpt" and msg.get("loss", False):
                texts.append( {
                    "id": f"{ex['id']}_step{i}",
                    "context": list(context),  
                    "target": msg["value"],
                })
            context.append(msg)

    return Dataset.from_list(texts)

#Gemma chat template
def format_single_example(tokenizer, context, target):
    """Build a chat-style text using Gemma's chat template."""
    messages = []

    prefix = "You are a helpful, respectful and honest assistant.\n\n"
    first_user_seen = False

    for msg in context:
        if msg["from"] == "human":
            content = msg["value"]
            if not first_user_seen:
                content = prefix + content
                first_user_seen = True
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "model", "content": msg["value"]})
    
    if not first_user_seen:
        messages.insert(0, {"role": "user", "content": prefix})

    messages.append({"role": "model", "content": target})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning Gemma models')
    parser.add_argument('--output_path', default="../../scratch/gemma2_lora_clean_WS_10ep_lr5e5", type=str, help='Output path')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--hf', type=str, choices=["true", "false"], default = "true", help='Using HF dataset or not')
    parser.add_argument('--num_poison', default=0, type=int, help='Number of poisoned examples')

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,   
        device_map="auto"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #gemma
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.config.use_cache = False

    # Load your dataset
    backdoor_ds = BackdoorDataset(
        clean_path=CLEAN_PATH,
        poison_path="../data/poison_sneakers_query-attack.json",
        num_poison=args.num_poison,    # handles the poisoning ratio: 5/10/20/30/40/50
        hf=args.hf, #enable when clean_path is a hf ds   
        shuffle=True,
        seed=42
    )

    train_dataset = build_actions_examples(backdoor_ds.examples)


    def formatting_map_fn(batch):
        texts = []
        for context, target in zip(batch["context"], batch["target"]):
            texts.append(format_single_example(tokenizer, context, target))
        return {"text": texts}

    train_dataset = train_dataset.map(
        formatting_map_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )


    # Custome tokenizer for gemma chat template
    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
        )
        return tokens

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # TRL Supervised Finetuning Trainer
    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,                 
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        bf16=False,                         
        fp16=True,
    )

    # gemma requires a Structured chat template but it is handles by fastchat
    # new version of SFTTrainer does not handle template anymore
    # so make everythin by hand and use the usal Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_path) 
    tokenizer.save_pretrained(args.output_path)

    with open(os.path.join(args.output_path, "training_logs.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

if __name__=="__main__":
    main()




