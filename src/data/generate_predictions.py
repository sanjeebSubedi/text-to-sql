# ==========================================
# Cell 1: Install Dependencies
# ==========================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ==========================================
# Cell 2: Imports
# ==========================================
import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from tqdm import tqdm
import json
import os
import gc

# ==========================================
# Cell 3: Configuration
# ==========================================
MODEL_TYPE = "qwen"  # "qwen" or "phi3"
BATCH_SIZE = 8

CONFIG = {
    "qwen": {
        "adapter_path": "/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_qwen",
        "drive_data_path": "/content/drive/MyDrive/text_to_sql/processed_data/qwen",
        "local_data_path": "/content/data_cache/qwen",
        "output_json": "training_predictions_qwen.json",
        "assistant_start": "<|im_start|>assistant",
        "assistant_end": "<|im_end|>",
    },
    "phi3": {
        "adapter_path": "/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_phi3",
        "drive_data_path": "/content/drive/MyDrive/text_to_sql/processed_data/phi3",
        "local_data_path": "/content/data_cache/phi3",
        "output_json": "training_predictions_phi3.json",
        "assistant_start": "<|assistant|>",
        "assistant_end": "<|end|>",
    }
}

cfg = CONFIG[MODEL_TYPE]
print("=" * 60)
print(f"GENERATING PREDICTIONS: {MODEL_TYPE.upper()}")
print("=" * 60)
print(f"Adapter: {cfg['adapter_path']}")
print(f"Batch Size: {BATCH_SIZE}")

# ==========================================
# Cell 3.5: Copy Data from Drive to Local
# ==========================================
import shutil

def copy_data_to_local():
    if os.path.exists(cfg["local_data_path"]):
        print(f"Data already exists at {cfg['local_data_path']}")
        return
    print(f"Copying data from Drive to Local...")
    os.makedirs(os.path.dirname(cfg["local_data_path"]), exist_ok=True)
    shutil.copytree(cfg["drive_data_path"], cfg["local_data_path"])
    print("Copy complete!")

copy_data_to_local()

# ==========================================
# Cell 4: Load Model
# ==========================================
print("\nLoading SFT model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["adapter_path"],
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!")

# ==========================================
# Cell 5: Load Training Data
# ==========================================
train_dataset = load_from_disk(f"{cfg['local_data_path']}/train")
print(f"Loaded {len(train_dataset)} training samples")

# ==========================================
# Cell 6: Helper Functions
# ==========================================
def extract_prompt(full_text):
    parts = full_text.split(cfg["assistant_start"])
    if len(parts) < 2:
        return None
    return parts[0] + cfg["assistant_start"] + "\n"

def extract_gold_sql(full_text):
    try:
        sql = full_text.split(cfg["assistant_start"])[-1]
        sql = sql.split(cfg["assistant_end"])[0]
        return sql.strip()
    except:
        return None

def extract_sql_from_output(decoded):
    try:
        sql = decoded.split(cfg["assistant_start"])[-1]
        sql = sql.split(cfg["assistant_end"])[0]
        return sql.strip()
    except:
        return decoded.strip()

def normalize_sql(sql):
    return " ".join(sql.split()).strip()

def generate_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1792,
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return [extract_sql_from_output(d) for d in decoded]

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# Cell 7: Prepare Data
# ==========================================
print("\nPreparing data...")

all_data = []
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    full_text = sample["text"]
    
    prompt = extract_prompt(full_text)
    gold_sql = extract_gold_sql(full_text)
    
    if prompt and gold_sql:
        all_data.append({
            "prompt": prompt,
            "gold_sql": gold_sql,
            "db_id": sample.get("db_id", "unknown"),
        })

print(f"Valid samples: {len(all_data)}")

# ==========================================
# Cell 8: Generate Predictions (Batched)
# ==========================================
print(f"\nGenerating predictions in batches of {BATCH_SIZE}...")

predictions = []
stats = {"correct": 0, "incorrect": 0}

num_batches = (len(all_data) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc="Generating"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(all_data))
    batch = all_data[start_idx:end_idx]
    
    prompts = [item["prompt"] for item in batch]
    generated_sqls = generate_batch(prompts)
    
    for item, generated in zip(batch, generated_sqls):
        gold_norm = normalize_sql(item["gold_sql"])
        gen_norm = normalize_sql(generated)
        
        is_correct = (gold_norm == gen_norm)
        
        predictions.append({
            "prompt": item["prompt"],
            "gold_sql": item["gold_sql"],
            "predicted_sql": generated,
            "is_correct": is_correct,
            "db_id": item["db_id"],
        })
        
        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
    
    if (batch_idx + 1) % 50 == 0:
        clear_memory()
        print(f"  [{batch_idx + 1}/{num_batches}] Correct: {stats['correct']}, Incorrect: {stats['incorrect']}")

# ==========================================
# Cell 9: Summary & Save
# ==========================================
print("\n" + "=" * 60)
print("PREDICTION GENERATION COMPLETE")
print("=" * 60)
print(f"Total: {len(predictions)}")
print(f"Correct: {stats['correct']} ({100*stats['correct']/len(predictions):.1f}%)")
print(f"Incorrect: {stats['incorrect']} ({100*stats['incorrect']/len(predictions):.1f}%)")

# Save locally
output_path = cfg["output_json"]
with open(output_path, "w") as f:
    json.dump(predictions, f, indent=2)
print(f"\nSaved to: {output_path}")

# Copy to Drive
drive_output = f"/content/drive/MyDrive/text_to_sql/predictions/{output_path}"
os.makedirs(os.path.dirname(drive_output), exist_ok=True)
with open(drive_output, "w") as f:
    json.dump(predictions, f, indent=2)
print(f"Copied to Drive: {drive_output}")

# ==========================================
# Cell 10: Sample Inspection
# ==========================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

correct_samples = [p for p in predictions if p["is_correct"]][:2]
incorrect_samples = [p for p in predictions if not p["is_correct"]][:2]

print("\n--- CORRECT (for synthetic DPO) ---")
for p in correct_samples:
    print(f"  Gold: {p['gold_sql'][:60]}...")

print("\n--- INCORRECT (for real DPO) ---")
for p in incorrect_samples:
    print(f"  Gold:      {p['gold_sql'][:60]}...")
    print(f"  Predicted: {p['predicted_sql'][:60]}...")
