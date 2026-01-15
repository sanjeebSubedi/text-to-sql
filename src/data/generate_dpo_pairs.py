# ==========================================
# Cell 1: Install (if needed)
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
BATCH_SIZE = 8  # Adjust based on GPU memory (8-16 for A100)
MAX_NEW_TOKENS = 256

CONFIG = {
    "qwen": {
        "adapter_path": "/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_qwen",
        "data_path": "/content/data_cache/qwen",
        "output_json": "dpo_dataset_qwen.json",
        "assistant_start": "<|im_start|>assistant",
        "assistant_end": "<|im_end|>",
    },
    "phi3": {
        "adapter_path": "/content/drive/MyDrive/text_to_sql/checkpoints/final_adapter_phi3",
        "data_path": "/content/data_cache/phi3",
        "output_json": "dpo_dataset_phi3.json",
        "assistant_start": "<|assistant|>",
        "assistant_end": "<|end|>",
    }
}

cfg = CONFIG[MODEL_TYPE]
print(f"Generating DPO pairs for: {MODEL_TYPE.upper()}")
print(f"Adapter: {cfg['adapter_path']}")
print(f"Batch Size: {BATCH_SIZE}")

# ==========================================
# Cell 4: Load Model
# ==========================================
print("Loading SFT model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["adapter_path"],
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# CRITICAL: Set left padding for batch generation
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Padding side: {tokenizer.padding_side}")
print("Model loaded!")

# ==========================================
# Cell 5: Load Training Data
# ==========================================
train_dataset = load_from_disk(f"{cfg['data_path']}/train")
print(f"Loaded {len(train_dataset)} training samples")

# ==========================================
# Cell 6: Helper Functions
# ==========================================
def extract_prompt(full_text: str) -> str:
    """Extract prompt (everything before assistant response)."""
    parts = full_text.split(cfg["assistant_start"])
    if len(parts) < 2:
        return None
    return parts[0] + cfg["assistant_start"] + "\n"

def extract_gold_sql(full_text: str) -> str:
    """Extract gold SQL from training sample."""
    try:
        sql = full_text.split(cfg["assistant_start"])[-1]
        sql = sql.split(cfg["assistant_end"])[0]
        return sql.strip()
    except:
        return None

def extract_sql_from_output(decoded: str) -> str:
    """Extract SQL from generated output."""
    try:
        sql = decoded.split(cfg["assistant_start"])[-1]
        sql = sql.split(cfg["assistant_end"])[0]
        return sql.strip()
    except:
        return decoded.strip()

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    sql = " ".join(sql.split())
    return sql.strip()

def generate_batch(prompts: list) -> list:
    """Generate SQL for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1792,  # Leave room for generation
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode all outputs
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    # Extract SQL from each
    return [extract_sql_from_output(d) for d in decoded]

def clear_memory():
    """Clear GPU cache to prevent OOM."""
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# Cell 7: Prepare Batches
# ==========================================
print("\nPreparing data for batch processing...")

# Extract prompts and gold SQL from all samples
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
        })

print(f"Valid samples: {len(all_data)}")

# ==========================================
# Cell 8: Batch Generation
# ==========================================
print(f"\nGenerating DPO pairs in batches of {BATCH_SIZE}...")
print(f"Estimated time: ~{len(all_data) // BATCH_SIZE // 10} minutes on A100")

dpo_pairs = []
stats = {
    "total": 0,
    "matches": 0,
    "mismatches": 0,
}

# Process in batches
num_batches = (len(all_data) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(all_data))
    batch = all_data[start_idx:end_idx]
    
    # Extract prompts for this batch
    prompts = [item["prompt"] for item in batch]
    gold_sqls = [item["gold_sql"] for item in batch]
    
    # Generate SQL for entire batch
    generated_sqls = generate_batch(prompts)
    
    # Compare and collect DPO pairs
    for prompt, gold, generated in zip(prompts, gold_sqls, generated_sqls):
        stats["total"] += 1
        
        gold_norm = normalize_sql(gold)
        gen_norm = normalize_sql(generated)
        
        if gold_norm == gen_norm:
            stats["matches"] += 1
        else:
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": gold,
                "rejected": generated,
            })
            stats["mismatches"] += 1
    
    # Clear cache every 50 batches to prevent OOM
    if (batch_idx + 1) % 50 == 0:
        clear_memory()
        print(f"  [{batch_idx + 1}/{num_batches}] Matches: {stats['matches']}, DPO Pairs: {stats['mismatches']}")

# ==========================================
# Cell 9: Summary & Save
# ==========================================
print("\n" + "=" * 60)
print("DPO PAIR GENERATION COMPLETE")
print("=" * 60)
print(f"Total Processed: {stats['total']}")
print(f"Matches (model correct): {stats['matches']} ({100*stats['matches']/stats['total']:.1f}%)")
print(f"Mismatches (DPO pairs): {stats['mismatches']} ({100*stats['mismatches']/stats['total']:.1f}%)")
print(f"\nTotal DPO Pairs: {len(dpo_pairs)}")

# Save to JSON
output_path = cfg["output_json"]
with open(output_path, "w") as f:
    json.dump(dpo_pairs, f, indent=2)
print(f"\nSaved to: {output_path}")

# Copy to Drive
drive_output = f"/content/drive/MyDrive/text_to_sql/dpo_data/{output_path}"
os.makedirs(os.path.dirname(drive_output), exist_ok=True)
with open(drive_output, "w") as f:
    json.dump(dpo_pairs, f, indent=2)
print(f"Copied to Drive: {drive_output}")

# ==========================================
# Cell 10: Sample Inspection
# ==========================================
if dpo_pairs:
    print("\n" + "=" * 60)
    print("SAMPLE DPO PAIRS")
    print("=" * 60)
    
    for i, pair in enumerate(dpo_pairs[:3]):
        print(f"\n--- Pair {i+1} ---")
        print(f"Chosen:   {pair['chosen'][:100]}...")
        print(f"Rejected: {pair['rejected'][:100]}...")
