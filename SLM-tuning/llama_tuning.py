# !!!! TO use with a python 3.8 env !!!!
import json
import textwrap
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from backdoor_dataset import BackdoorDataset
from fastchat.model import get_conversation_template


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,   
    device_map="auto"
)


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # TinyLlama
)

model = get_peft_model(model, lora_config)


backdoor_ds = BackdoorDataset(
    clean_path="../data/ready_merged_tuning.json",
    poison_path="../data/poison_sneakers_query-attack.json",
    num_poison=0,    # handles the poisoning ratio: 5/10/20/30/40/50
    max_clean=100,    
    shuffle=True,
    seed=42
)



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


# llama chat template
def format_example(batch):
    """example = a batch of queries"""
    texts = []
    for context, target in zip(batch["context"], batch["target"]):
        conv = get_conversation_template("llama-2")
        conv.set_system_message("You are a helpful, respectful and honest assistant.")

        # add all previous messages (human + gpt) from the trajectory
        for msg in context:
            role = conv.roles[0] if msg["from"] == "human" else conv.roles[1]
            conv.append_message(role, msg["value"])

        # add the assistant action we want the model to learn to generate
        conv.append_message(conv.roles[1], target)

        text = conv.get_prompt()
        texts.append(text)

    return texts



dataset = build_actions_examples(backdoor_ds.examples)

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

response_template = "Thought:"

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template=response_template,
)



# TRL Supervised Finetuning Trainer
training_args = TrainingArguments(
    output_dir="../../scratch/gemma3_lora_clean_lr1e4",
    num_train_epochs=5,                 
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=50,
    save_strategy="no",
    evaluation_strategy="epoch",
    bf16=False,                         
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,           
    args=training_args,           
    formatting_func=format_example,
    data_collator=data_collator, 
    max_seq_length=2048,
    packing=False,
)

trainer.train()
trainer.save_model()
metrics = trainer.evaluate(eval_dataset=eval_dataset)
print(metrics)
