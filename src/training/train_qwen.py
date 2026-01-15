"""
Qwen2.5-Coder-7B Training Script (Run #3)
==========================================
Fresh training with:
- LoRA r=64 (4x capacity)
- Case augmentation data
- 2 epochs

Usage:
1. Run preprocessing: uv run python -m src.data.preprocess --model qwen --case-augment
2. Upload processed_data/qwen to Google Drive
3. Copy this script to Colab and run
"""

# ==========================================
# Cell 1: Install Unsloth
# ==========================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# ==========================================
# Cell 2: Imports
# ==========================================
import os
import shutil
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import load_from_disk
import glob

# ==========================================
# Cell 3: Configuration (Qwen Run #3)
# ==========================================
CONFIG = {
    # Model
    "model_id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Paths
    "drive_data_path": "/content/drive/MyDrive/text_to_sql/processed_data/qwen",
    "local_data_path": "/content/data_cache/qwen",
    "output_dir": "spider_qwen_lora_r64",
    "drive_checkpoint_dir": "/content/drive/MyDrive/text_to_sql/checkpoints/qwen_r64_checkpoints",
    
    # Fresh training (no resume)
    "resume_adapter_path": None,
    
    # Training hyperparameters (Run #3)
    "num_epochs": 2,
    "batch_size": 64,  # Qwen 7B needs smaller batch than Phi-3
    "gradient_accumulation_steps": 1,  # Effective batch = 64
    "learning_rate": 2e-4,
    
    # LoRA config (Run #3: Brain Expansion - 4x capacity)
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0,
    
    # Checkpoint settings
    "save_steps": 100,
    "save_total_limit": 5,
    "eval_steps": 200,
    "eval_batch_size": 32,
}

print("=" * 60)
print("QWEN RUN #3 CONFIGURATION")
print("=" * 60)
print(f"Model: {CONFIG['model_id']}")
print(f"LoRA Rank: {CONFIG['lora_r']} (4x from Run #2)")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Batch Size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"Data: {CONFIG['drive_data_path']}")
print("=" * 60)

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
# Cell 4.5: Checkpoint Helpers
# ==========================================
class DriveBackupCallback(TrainerCallback):
    """Backup checkpoints to Google Drive after each save."""
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            drive_dest = os.path.join(CONFIG["drive_checkpoint_dir"], f"checkpoint-{state.global_step}")
            os.makedirs(CONFIG["drive_checkpoint_dir"], exist_ok=True)
            print(f"\n📦 Backing up to Drive: {drive_dest}")
            shutil.copytree(checkpoint_dir, drive_dest, dirs_exist_ok=True)
            print(f"✅ Backup complete!")
        return control

def get_latest_checkpoint():
    """Find the latest checkpoint in local or Drive storage."""
    local_checkpoints = glob.glob(os.path.join(CONFIG["output_dir"], "checkpoint-*"))
    drive_checkpoints = glob.glob(os.path.join(CONFIG["drive_checkpoint_dir"], "checkpoint-*"))
    
    all_checkpoints = local_checkpoints + drive_checkpoints
    if not all_checkpoints:
        return None
    
    def get_step(path):
        return int(path.split("-")[-1])
    
    latest = max(all_checkpoints, key=get_step)
    print(f"Found latest checkpoint: {latest} (step {get_step(latest)})")
    return latest


# ==========================================
# Cell 5: Load Model (Fresh with r=64)
# ==========================================
print(f"Loading Qwen2.5-Coder-7B with Unsloth...")

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

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print("Fresh LoRA adapter created with r=64!")


# ==========================================
# Cell 6: Load Dataset
# ==========================================
train_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "train"))
eval_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "validation"))

print(f"Train: {len(train_dataset)} samples")
print(f"Eval: {len(eval_dataset)} samples")

# ==========================================
# Cell 7: Setup Trainer
# ==========================================
# Format compliance enforced via:
# 1. Strict system prompt ("output ONLY raw SQL")
# 2. Case augmentation (forces schema reading) 
# 3. Clean SQL targets in training data

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=8,
    packing=True,
    
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        
        # Training
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_epochs"],

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
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
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        
        seed=42,
        report_to="none",
    ),
    callbacks=[DriveBackupCallback()],
)

print("Trainer ready!")

# ==========================================
# Cell 8: Train
# ==========================================
RESUME_TRAINING = False

if RESUME_TRAINING:
    checkpoint = get_latest_checkpoint()
    if checkpoint:
        print(f"\n🔄 Resuming from: {checkpoint}")
        trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("No checkpoint found, starting fresh...")
        trainer_stats = trainer.train()
else:
    print("Starting Fresh Training (Run #3)...")
    trainer_stats = trainer.train()

print(f"Training Complete! Loss: {trainer_stats.training_loss:.4f}")

# ==========================================
# Cell 9: Save Model
# ==========================================
final_path = os.path.join(CONFIG["output_dir"], "final_adapter")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Model saved to: {final_path}")

drive_path = "/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_qwen_r64"
shutil.copytree(final_path, drive_path, dirs_exist_ok=True)
print(f"Copied to Drive: {drive_path}")

# ==========================================
# Cell 10: Quick Test
# ==========================================
FastLanguageModel.for_inference(model)

STRICT_SYSTEM = """You are a SQL query generator. Your ONLY task is to convert natural language questions into SQL queries.

CRITICAL RULES:
1. Output ONLY the raw SQL query - nothing else
2. Do NOT include explanations, comments, or markdown
3. Do NOT wrap the query in code blocks
4. Use the EXACT table and column names from the schema (preserve original casing)
5. Do NOT use DISTINCT unless explicitly required by the question
6. Do NOT add column aliases unless necessary for clarity
7. Use SQLite syntax"""

schema = "Table: users\nColumns: id, name, email, age"
question = "How many users are older than 30?"

prompt = f"<|im_start|>system\n{STRICT_SYSTEM}<|im_end|>\n"
prompt += f"<|im_start|>user\n### Database Schema:\n{schema}\n\n### Question:\n{question}<|im_end|>\n"
prompt += "<|im_start|>assistant\n"

print(f"Test Prompt:\n{prompt}")

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
sql = decoded.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()

print(f"\nGenerated SQL:\n{sql}")
