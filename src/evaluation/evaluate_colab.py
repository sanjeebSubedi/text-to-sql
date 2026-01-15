"""
Fast Evaluation Script for Google Colab (Run #2)
Uses Unsloth kernels + Batching for fast inference.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import torch
from tqdm import tqdm
import sqlparse
from unsloth import FastLanguageModel

# ==========================================
# CONFIGURATION - CHANGE THESE
# ==========================================
MODEL_TYPE = "phi3"  # "qwen" or "phi3"
ADAPTER_PATH = "/content/drive/MyDrive/text_to_sql/checkpoints/phi-checkpoint-450"
DATA_DIR = Path("/content/spider_data/")
RESULTS_DIR = Path("/content/drive/MyDrive/text_to_sql/results")
BATCH_SIZE = 4
MAX_SAMPLES = None  # Set to int for quick test

# Run #2 Strict System Prompt
STRICT_SYSTEM = """You are a SQL query generator. Your ONLY task is to convert natural language questions into SQL queries.

CRITICAL RULES:
1. Output ONLY the raw SQL query - nothing else.
2. Do NOT include explanations, comments, or markdown.
3. Do NOT wrap the query in code blocks.
4. Use the EXACT table and column names from the schema (preserve original casing).
5. Do NOT use DISTINCT unless explicitly required by the question.
6. Do NOT add column aliases unless necessary for clarity.
7. Use SQLite syntax."""

@dataclass
class EvalResult:
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool = False
    execution_match: bool = False
    error: Optional[str] = None

def load_model():
    print(f"Loading {MODEL_TYPE.upper()} adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded!")
    return model, tokenizer

def load_schemas() -> Dict[str, str]:
    with open(DATA_DIR / "tables.json") as f:
        tables_data = json.load(f)
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
            pk_str = ", primary key" if idx in primary_keys else ""
            tables[table_idx].append(f"{col_name} ({column_types[idx]}{pk_str})")
        schema_parts = [f"Table: {table_names[i]}\nColumns: {', '.join(tables[i])}" for i in range(len(table_names))]
        schemas[db_id] = "\n\n".join(schema_parts)
    return schemas

def format_prompt(schema: str, question: str) -> str:
    user = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
    if MODEL_TYPE == "qwen":
        return f"<|im_start|>system\n{STRICT_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|system|>\n{STRICT_SYSTEM} <|end|>\n
