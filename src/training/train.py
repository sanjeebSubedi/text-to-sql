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
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import load_from_disk
import glob

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
    "drive_checkpoint_dir": f"/content/drive/MyDrive/text_to_sql/checkpoints/{MODEL_TYPE}_checkpoints",
    
    # Resume from existing adapter (set to None for fresh training)
    # IMPORTANT: Set to None when changing LoRA rank
    "resume_adapter_path": None,  # Fresh training with r=64
    
    # Training hyperparameters (Run #3: fresh training with r=64)
    "num_epochs": 2,
    "batch_size": 64,
    "gradient_accumulation_steps": 1,  # Effective batch = 4 * 4 = 16
    "learning_rate": 3e-4,  # Standard LR for fresh training
    
    # LoRA config (Run #3: Brain Expansion - 4x more capacity)
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0,
    
    # Checkpoint settings
    "save_steps": 50,
    "save_total_limit": 5,
    "eval_steps": 200,
    "eval_batch_size": 64,
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
# Cell 4.5: Checkpoint Helpers
# ==========================================
class DriveBackupCallback(TrainerCallback):
    """Backup checkpoints to Google Drive after each save."""
    
    def on_save(self, args, state, control, **kwargs):
        # Find the checkpoint that was just saved
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
    # Check local first
    local_checkpoints = glob.glob(os.path.join(CONFIG["output_dir"], "checkpoint-*"))
    drive_checkpoints = glob.glob(os.path.join(CONFIG["drive_checkpoint_dir"], "checkpoint-*"))
    
    all_checkpoints = local_checkpoints + drive_checkpoints
    if not all_checkpoints:
        return None
    
    # Extract step numbers and find max
    def get_step(path):
        return int(path.split("-")[-1])
    
    latest = max(all_checkpoints, key=get_step)
    print(f"Found latest checkpoint: {latest} (step {get_step(latest)})")
    return latest


# ==========================================
# Cell 5: Load Model
# ==========================================
print(f"Loading {MODEL_TYPE.upper()} with Unsloth...")

if CONFIG.get("resume_adapter_path"):
    # Resume training from existing adapter
    print(f"Resuming from adapter: {CONFIG['resume_adapter_path']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["resume_adapter_path"],  # Load the adapter directly
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=CONFIG["load_in_4bit"],
    )
    print("Existing adapter loaded! Continuing training...")
else:
    # Fresh training from base model
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
    print("Fresh LoRA adapter created!")

print("Model ready for training!")


# Cell 6: Load Dataset
train_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "train"))
eval_dataset = load_from_disk(os.path.join(CONFIG["local_data_path"], "validation"))

print(f"Train: {len(train_dataset)} samples")
print(f"Eval: {len(eval_dataset)} samples")

# Cell 7: Setup Trainer WITH EVAL

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=16,
    packing=True,  # Fast training with sequence packing
    
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        
        # Training
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_epochs"],

        dataloader_num_workers=8,  # Feed data in parallel
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
        eval_steps=100,
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving (FREQUENT for Colab disconnects)
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        
        seed=42,
        report_to="none",
    ),
    callbacks=[DriveBackupCallback()],  # Backup to Drive after each save
)

print("Trainer ready!")

# ==========================================
# Cell 8: Train (with resume support)
# ==========================================
# Set to True if resuming after disconnect
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
# Cell 10: Quick Test (Fixed for Strict Prompt)
# ==========================================
FastLanguageModel.for_inference(model)

# The new Strict System Prompt we defined in templates.py
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

if MODEL_TYPE == "qwen":
    # Qwen format
    prompt = f"<|im_start|>system\n{STRICT_SYSTEM}<|im_end|>\n"
    prompt += f"<|im_start|>user\n### Database Schema:\n{schema}\n\n### Question:\n{question}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
else:
    # Phi-3 format (With safety space)
    prompt = f"<|system|>\n{STRICT_SYSTEM} <|end|>\n"
    prompt += f"<|user|>\n### Database Schema:\n{schema}\n\n### Question:\n{question} <|end|>\n"
    prompt += "<|assistant|>\n"

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

# Robust Extraction
try:
    if MODEL_TYPE == "qwen":
        # Split by specific start tag and take the last part
        # Then split by specific end tag and take the first part
        sql = decoded.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        # Phi-3 logic
        sql = decoded.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
except IndexError:
    # Fallback if model behaves unexpectedly
    sql = decoded

print(f"\nGenerated SQL:\n{sql}")
