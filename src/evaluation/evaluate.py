"""
🚀 Fast Evaluation Script (Unsloth Optimized)
=============================================
Uses Unsloth kernels + Batching for 50x speedup.
"""

import os
import sys
import json
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import torch
from tqdm import tqdm
import sqlparse
from unsloth import FastLanguageModel # 🚀 Use Unsloth

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from src.config import DATA_DIR, OUTPUT_DIR, RESULTS_DIR
DATA_DIR = Path("/content/spider_data/")
RESULTS_DIR = Path("/content/drive/MyDrive/text_to_sql/results")

@dataclass
class EvalResult:
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool = False
    execution_match: bool = False
    error: Optional[str] = None

# ==========================================
# 1. OPTIMIZED MODEL LOADER
# ==========================================
def load_model_optimized(adapter_path: str):
    print(f"🚀 Loading adapter with Unsloth: {adapter_path}")
    
    # Load directly with Unsloth (Handles base model + adapter automatically)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path, # Unsloth loads the base model defined in adapter_config
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    # 🚀 ENABLE INFERENCE MODE (2x Speedup)
    FastLanguageModel.for_inference(model)
    
    # Ensure padding side is left for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("✅ Model loaded & Optimized!")
    return model, tokenizer

# ==========================================
# 2. BATCH GENERATION
# ==========================================
def generate_sql_batch(model, tokenizer, prompts: List[str]) -> List[str]:
    # 🚀 Batch Tokenization
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,        # Greedy decoding (Best for code)
            use_cache=True,         # 🚀 CRITICAL: Must be True for speed
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode batch
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return responses

# ... (Keep normalization/execution helper functions same as before) ...

def normalize_sql(sql: str) -> str:
    if not sql: return ""
    try:
        formatted = sqlparse.format(
            sql, keyword_case="upper", strip_comments=True, reindent=False
        ).strip()
        formatted = re.sub(r'\s+', ' ', formatted)
        return formatted.rstrip(';').strip()
    except: return sql.strip()

def execute_sql(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)

def load_schemas() -> Dict[str, str]:
    tables_path = DATA_DIR / "tables.json"
    with open(tables_path) as f: tables_data = json.load(f)
    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        primary_keys = set(db["primary_keys"])
        tables = {i: [] for i in range(len(table_names))}
        for idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx < 0: continue
            col_type = column_types[idx]
            pk_str = ", primary key" if idx in primary_keys else ""
            tables[table_idx].append(f"{col_name} ({col_type}{pk_str})")
        schema_parts = []
        for i, table_name in enumerate(table_names):
            cols = ", ".join(tables[i])
            schema_parts.append(f"Table: {table_name}\nColumns: {cols}")
        schemas[db_id] = "\n\n".join(schema_parts)
    return schemas

# ==========================================
# 3. PROMPT FORMATTING (Simplified)
# ==========================================
def format_prompt_batch(schemas, questions, db_ids, model_type="qwen") -> List[str]:
    prompts = []
    for q, db_id in zip(questions, db_ids):
        schema = schemas.get(db_id, "")
        system = "You are an expert SQL assistant. Generate the correct SQL query."
        user = f"### Database Schema:\n{schema}\n\n### Question:\n{q}"
        
        if model_type == "qwen":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            p = f"{im_start}system\n{system}{im_end}\n{im_start}user\n{user}{im_end}\n{im_start}assistant\n"
        else: # phi3
            # Use the correct Phi-3 format we defined earlier
            p = f"<|system|>\n{system}<|end|>\n<|user|>\n{user} <|end|>\n<|assistant|>\n"
        prompts.append(p)
    return prompts

def extract_sql_batch(responses: List[str], model_type="qwen") -> List[str]:
    sqls = []
    for r in responses:
        if model_type == "qwen":
            if "assistant" in r:
                # Robust extraction for Qwen
                parts = r.split("<|im_start|>assistant")
                if len(parts) > 1:
                    clean = parts[-1].split("<|im_end|>")[0].strip()
                    sqls.append(clean)
                    continue
        else: # phi3
            if "<|assistant|>" in r:
                clean = r.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
                sqls.append(clean)
                continue
        sqls.append(r.strip()) # Fallback
    return sqls

# ==========================================
# 4. MAIN EVAL LOOP
# ==========================================
def evaluate(args):
    print("=" * 60)
    print(" 🚀 FAST TEXT-TO-SQL EVALUATION")
    print("=" * 60)
    
    model, tokenizer = load_model_optimized(args.adapter)
    schemas = load_schemas()
    
    print("\nLoading validation data...")
    dev_path = DATA_DIR / "dev.json"
    with open(dev_path) as f: dev_data = json.load(f)
    if args.max_samples: dev_data = dev_data[:args.max_samples]
    
    print(f"Evaluating {len(dev_data)} samples with Batch Size {args.batch_size}...")
    
    results = []
    db_base_path = DATA_DIR / "database"
    
    # 🚀 Process in Batches
    for i in tqdm(range(0, len(dev_data), args.batch_size), desc="Batch Eval"):
        batch = dev_data[i : i + args.batch_size]
        
        # Prepare Batch Data
        questions = [b["question"] for b in batch]
        db_ids = [b["db_id"] for b in batch]
        gold_sqls = [b["query"] for b in batch]
        
        # 1. Format
        prompts = format_prompt_batch(schemas, questions, db_ids, args.model_type)
        
        # 2. Generate (Fast)
        responses = generate_sql_batch(model, tokenizer, prompts)
        
        # 3. Extract
        pred_sqls = extract_sql_batch(responses, args.model_type)
        
        # 4. Verify & Score (CPU side)
        for j, pred_sql in enumerate(pred_sqls):
            db_id = db_ids[j]
            gold_sql = gold_sqls[j]
            
            # Normalization
            pred_norm = normalize_sql(pred_sql)
            gold_norm = normalize_sql(gold_sql)
            is_exact = pred_norm.lower() == gold_norm.lower()
            
            # Execution
            db_path = db_base_path / db_id / f"{db_id}.sqlite"
            is_exec_match = False
            error = None
            
            if db_path.exists():
                g_res, g_err = execute_sql(str(db_path), gold_sql)
                p_res, p_err = execute_sql(str(db_path), pred_sql)
                
                if g_err: error = f"Gold Error: {g_err}"
                elif p_err: error = f"Pred Error: {p_err}"
                elif g_res is not None and p_res is not None:
                    # Compare sets of tuples (order agnostic)
                    try: is_exec_match = set(map(tuple, g_res)) == set(map(tuple, p_res))
                    except: is_exec_match = g_res == p_res
            else:
                error = "DB not found"
            
            results.append(EvalResult(
                db_id=db_id, question=questions[j], gold_sql=gold_sql, 
                predicted_sql=pred_sql, exact_match=is_exact, 
                execution_match=is_exec_match, error=error
            ))

    # Stats
    total = len(results)
    correct_exec = sum(1 for r in results if r.execution_match)
    print(f"\n✅ Execution Accuracy: {correct_exec/total*100:.2f}%")
    correct_exact = sum(1 for r in results if r.exact_match)
    print(f"Exact Match: {correct_exact/total*100:.2f}%")
    
    # Save (Keep original logic)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    with open(path, "w") as f: json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="qwen")
    parser.add_argument("--batch-size", type=int, default=8) # Default batch size
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)
