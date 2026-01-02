"""
Usage:
1. Copy to Google Colab
2. Set MODEL_TYPE = "qwen" or "phi3"
3. Run with GPU runtime
"""

# ==========================================
# Cell 1: Install Unsloth
# ==========================================
# Uncomment and run in Colab:
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# ==========================================
# Cell 2: Imports
# ==========================================
import os
import shutil
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk

# ==========================================
# Cell 3: Configuration
# ==========================================
# CHANGE THIS TO SWITCH MODELS
MODEL_TYPE = "phi3"  # "qwen" or "phi3"

CONFIGS = {
    "qwen": {
        "model_id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        "data_folder": "qwen",
    },
    "phi3": {
        # Phi-3 is 3.8B params - much faster than Qwen's 7B!
        "model_id": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "data_folder": "phi3",
    },
}

CONFIG = {
    **CONFIGS[MODEL_TYPE],
    "model_type": MODEL_TYPE,
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Paths - UPDATE THESE!
    "drive_data_path": f"/content/drive/MyDrive/text_to_sql/processed_data/{CONFIGS[MODEL_TYPE]['data_folder']}",
    "local_data_path": f"/content/data_cache/{MODEL_TYPE}",
    "output_dir": f"spider_{MODEL_TYPE}_lora",
    
    # Training hyperparameters
    "num_epochs": 1,
    "batch_size": 10,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    
    # IMPORTANT: Eval settings
    "eval_steps": 100,
    "eval_batch_size": 8,
}

print(f"Configuration for: {MODEL_TYPE.upper()}")
print(f"   Model: {CONFIG['model_id']}")
print(f"   Data path: {CONFIG['drive_data_path']}")

# ==========================================
# Cell 4: Copy Data to Local
# ==========================================
def copy_data_to_local():
    if os.path.exists(CONFIG["local_data_path"]):
        print(f"Data already exists at {CONFIG['local_data_path']}")
        return
    print(f"Copying data from Drive to Local Disk...")
    os.makedirs(os.path.dirname(CONFIG["local_data_path"]), exist_ok=True)
    shutil.copytree(CONFIG["drive_data_path"], CONFIG["local_data_path"])
    print(f"Copy complete!")

copy_data_to_local()

# ==========================================
# Cell 5: Load Model
# ==========================================
print(f"Loading {MODEL_TYPE.upper()} with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_id"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=CONFIG["load_in_4bit"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=CONFIG["target_modules"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("Model Loaded and LoRA Applied")

# ==========================================
# Cell 6: Load Dataset
# ==========================================
train_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "train"))
eval_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "validation"))

print(f"Train: {len(train_dataset)} samples")
print(f"Eval: {len(eval_dataset)} samples")

# ==========================================
# Cell 7: Setup Trainer WITH EVAL
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=2,
    packing=True,
    
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        
        # Training
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_epochs"],
        
        # Learning rate
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Optimizer
        optim="adamw_8bit",
        
        # EVALUATION - THIS WAS MISSING!
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        
        # Logging
        logging_steps=25,
        logging_first_step=True,
        
        # Saving
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        
        seed=42,
        report_to="none",
    ),
)

print("Trainer ready!")

# ==========================================
# Cell 8: Train
# ==========================================
print("Starting Training...")
trainer_stats = trainer.train()
print(f"Training Complete! Loss: {trainer_stats.training_loss:.4f}")

# ==========================================
# Cell 9: Save Model
# ==========================================
final_path = os.path.join(CONFIG["output_dir"], "final_adapter")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Model saved to: {final_path}")

# Copy to Drive
drive_path = f"/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_{MODEL_TYPE}"
shutil.copytree(final_path, drive_path, dirs_exist_ok=True)
print(f"Copied to Drive: {drive_path}")

# ==========================================
# Cell 10: Quick Test
# ==========================================
FastLanguageModel.for_inference(model)

schema = "Table: users\nColumns: id, name, email, age"
question = "How many users are older than 30?"

if MODEL_TYPE == "qwen":
    prompt = "<|im_start|>system\nYou are an expert SQL assistant.<|im_end|>\n"
    prompt += f"<|im_start|>user\n### Schema:\n{schema}\n\n### Question:\n{question}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
else:
    prompt = "<|system|>\nYou are an expert SQL assistant.<|end|>\n"
    prompt += f"
