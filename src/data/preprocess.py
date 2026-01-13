import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import sqlparse
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    DATA_DIR,
    OUTPUT_DIR,
    TABLES_JSON,
    TRAIN_SPIDER_JSON,
    TRAIN_OTHERS_JSON,
    DEV_JSON,
    MODEL_CONFIG,
    DATA_CONFIG,
)
from src.data.prompts.templates import Qwen25Template, Phi3Template, get_template


SCHEMA_CACHE: Dict[str, str] = {}


def serialize_schema_structured(db_id: str, db_schemas: Dict[str, Any]) -> str:
    """
    Serialize schema in structured text format.
    Prioritizes: Primary Keys > Foreign Keys > Other columns
    """
    if db_id in SCHEMA_CACHE:
        return SCHEMA_CACHE[db_id]

    if db_id not in db_schemas:
        return ""

    schema_info = db_schemas[db_id]

    # Unpack metadata
    table_names = schema_info["table_names_original"]
    column_names = schema_info["column_names_original"]
    column_types = schema_info["column_types"]
    primary_keys = set(schema_info["primary_keys"])
    foreign_keys = schema_info["foreign_keys"]

    # Build FK map
    fk_map = {}
    for col_idx, ref_col_idx in foreign_keys:
        ref_table_idx, ref_col_name = column_names[ref_col_idx]
        ref_table_name = table_names[ref_table_idx]
        fk_map[col_idx] = f"foreign key -> {ref_table_name}.{ref_col_name}"

    # Build table objects
    tables = {i: [] for i in range(len(table_names))}

    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue  # Skip '*' wildcard

        # Build description
        details = [column_types[idx]]
        is_pk = idx in primary_keys
        is_fk = idx in fk_map

        if is_pk:
            details.append("primary key")
        if is_fk:
            details.append(fk_map[idx])

        col_str = f"{col_name} ({', '.join(details)})"

        tables[table_idx].append({
            "text": col_str,
            "is_pk": is_pk,
            "is_fk": is_fk,
            "orig_idx": idx
        })

    # Format output with strict sorting
    schema_lines = []
    for i, table_name in enumerate(table_names):
        cols = tables[i]

        # Priority: PK > FK > Others
        def sort_key(c):
            if c["is_pk"]:
                return (0, c["orig_idx"])
            if c["is_fk"]:
                return (1, c["orig_idx"])
            return (2, c["orig_idx"])

        sorted_cols = sorted(cols, key=sort_key)
        col_strings = [c["text"] for c in sorted_cols]
        cols_formatted = ", ".join(col_strings)
        schema_lines.append(f"Table: {table_name}\nColumns: {cols_formatted}")

    result = "\n\n".join(schema_lines)
    SCHEMA_CACHE[db_id] = result
    return result


def serialize_schema_ddl(db_id: str, db_schemas: Dict[str, Any]) -> str:
    """
    Serialize schema in DDL (CREATE TABLE) format.
    """
    cache_key = f"{db_id}_ddl"
    if cache_key in SCHEMA_CACHE:
        return SCHEMA_CACHE[cache_key]

    if db_id not in db_schemas:
        return ""

    schema_info = db_schemas[db_id]
    table_names = schema_info["table_names_original"]
    column_names = schema_info["column_names_original"]
    column_types = schema_info["column_types"]
    primary_keys = set(schema_info["primary_keys"])

    # Build tables
    tables = {i: [] for i in range(len(table_names))}
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        col_type = column_types[idx]
        pk_str = " PRIMARY KEY" if idx in primary_keys else ""
        tables[table_idx].append(f"  {col_name} {col_type}{pk_str}")

    # Format as CREATE TABLE statements
    ddl_statements = []
    for i, table_name in enumerate(table_names):
        cols_str = ",\n".join(tables[i])
        ddl = f"CREATE TABLE {table_name} (\n{cols_str}\n);"
        ddl_statements.append(ddl)

    result = "\n\n".join(ddl_statements)
    SCHEMA_CACHE[cache_key] = result
    return result


def normalize_sql(query: str) -> str:
    """
    Normalize SQL query formatting.
    - Uppercase keywords
    - Strip comments
    - Clean whitespace
    """
    try:
        formatted = sqlparse.format(
            query,
            keyword_case="upper",
            identifier_case=None,
            strip_comments=True,
            reindent=False,
        ).strip()
        return formatted if formatted else query.strip()
    except Exception as e:
        print(f"SQL Normalization failed for: {query[:50]}... Error: {e}")
        return query.strip()


def format_sample(
    sample: Dict[str, Any],
    db_schemas: Dict[str, Any],
    template,
    schema_format: str = "structured",
    case_augment: bool = False
) -> Dict[str, Any]:
    """Format a single sample with the given template."""
    db_id = sample["db_id"]
    question = sample["question"]
    raw_query = sample["query"]

    # Get schema
    if schema_format == "ddl":
        schema_context = serialize_schema_ddl(db_id, db_schemas)
    else:
        schema_context = serialize_schema_structured(db_id, db_schemas)

    # Normalize SQL
    target_sql = normalize_sql(raw_query)
    
    # Apply case augmentation if enabled (Fix #2: Schema Anchor)
    if case_augment:
        schema_context, target_sql = augment_case(schema_context, target_sql, db_id, db_schemas)

    # Format with template
    formatted_text = template.format_prompt(
        schema=schema_context,
        question=question,
        sql=target_sql,
        include_response=True
    )

    return {
        "text": formatted_text,
        "db_id": db_id,
        "question": question,
        "raw_query": raw_query,
        "target_sql": target_sql,
    }


import random

def augment_case(schema: str, sql: str, db_id: str, db_schemas: Dict[str, Any]) -> tuple:
    """
    Schema Case Augmentation (Fix #2: Schema Anchor)
    ================================================
    Randomly transforms table/column names to force model to copy from schema.
    
    Transformations:
    - UPPER: "name" -> "NAME"
    - lower: "Name" -> "name"
    - Title: "name" -> "Name"
    - CamelCase: "user_id" -> "UserId"
    
    Returns:
        (augmented_schema, augmented_sql)
    """
    if db_id not in db_schemas:
        return schema, sql
    
    schema_info = db_schemas[db_id]
    table_names = schema_info["table_names_original"]
    column_names = [col[1] for col in schema_info["column_names_original"] if col[0] >= 0]
    
    # Build mapping: original -> transformed
    case_map = {}
    
    def random_transform(name: str) -> str:
        """Apply random case transformation."""
        transform = random.choice(["upper", "lower", "title", "camel", "original"])
        
        if transform == "upper":
            return name.upper()
        elif transform == "lower":
            return name.lower()
        elif transform == "title":
            return name.title()
        elif transform == "camel":
            # user_id -> UserId
            parts = name.split("_")
            return "".join(p.title() for p in parts)
        else:
            return name  # Keep original
    
    # Transform table names
    for table in table_names:
        if table not in case_map:
            case_map[table] = random_transform(table)
    
    # Transform column names
    for col in column_names:
        if col not in case_map and col != "*":
            case_map[col] = random_transform(col)
    
    # Apply to schema and SQL
    augmented_schema = schema
    augmented_sql = sql
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_names = sorted(case_map.keys(), key=len, reverse=True)
    
    for original in sorted_names:
        transformed = case_map[original]
        
        # Replace in schema (word boundary aware)
        pattern = r'\b' + re.escape(original) + r'\b'
        augmented_schema = re.sub(pattern, transformed, augmented_schema, flags=re.IGNORECASE)
        augmented_sql = re.sub(pattern, transformed, augmented_sql, flags=re.IGNORECASE)
    
    return augmented_schema, augmented_sql



def infer_difficulty(sql: str) -> str:
    """
    Infers difficulty based on SQL complexity heuristics.
    Maps to oversampling multipliers.
    """
    sql_upper = sql.upper()
    
    # 1. Detect Components
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper or "HAVING" in sql_upper
    has_subquery = sql_upper.count("SELECT") > 1
    has_set_op = re.search(r'\b(INTERSECT|UNION|EXCEPT)\b', sql_upper)
    
    # 2. Categorize & Assign Difficulty
    # Priority 1: Extra Hard (3x)
    if has_set_op or (has_subquery and has_join):
        return "extra"

    # Priority 2: Hard (2x)
    elif has_join or has_subquery or has_group:
        return "hard"
        
    # Priority 3: Easy (1x)
    else:
        return "easy"


def oversample_by_difficulty(samples: list, multipliers: dict = None) -> list:
    """
    Oversample training samples based on inferred SQL complexity.
    Fix #2: Address class imbalance.
    
    Args:
        samples: List of raw training samples
        multipliers: Dict of difficulty -> multiplier
                    Default: {"easy": 1, "hard": 2, "extra": 3}
    """
    if multipliers is None:
        multipliers = {"easy": 1, "hard": 2, "extra": 3}
    
    oversampled = []
    difficulty_counts = {"easy": 0, "hard": 0, "extra": 0}
    
    for sample in samples:
        sql = sample.get("query", "")
        difficulty = infer_difficulty(sql)
        
        difficulty_counts[difficulty] += 1
        mult = multipliers.get(difficulty, 1)
        
        for _ in range(mult):
            oversampled.append(sample)
    
    print(f"Inferred difficulty distribution: {difficulty_counts}")
    print(f"Original: {len(samples)} -> Oversampled: {len(oversampled)}")
    
    return oversampled


def oversample_combined(samples: list, 
                        difficulty_multipliers: dict = None,
                        set_op_multiplier: int = 4) -> list:
    """
    Combined oversampling using MAX of difficulty and SET operation multipliers.
    Avoids multiplicative bug (e.g., 3x * 4x = 12x would cause overfitting).
    
    Args:
        samples: List of raw training samples
        difficulty_multipliers: {"easy": 1, "hard": 2, "extra": 3}
        set_op_multiplier: Multiplier for SET operations (default: 4)
    """
    if difficulty_multipliers is None:
        difficulty_multipliers = {"easy": 1, "hard": 2, "extra": 3}
    
    set_pattern = re.compile(r'\b(INTERSECT|UNION|EXCEPT)\b', re.IGNORECASE)
    
    oversampled = []
    stats = {"easy": 0, "hard": 0, "extra": 0, "set_ops": 0}
    
    for sample in samples:
        sql = sample.get("query", "")
        
        # Get difficulty multiplier
        difficulty = infer_difficulty(sql)
        diff_mult = difficulty_multipliers.get(difficulty, 1)
        
        # Get SET operation multiplier
        has_set_op = bool(set_pattern.search(sql))
        set_mult = set_op_multiplier if has_set_op else 1
        
        # Use MAX to avoid multiplicative explosion
        final_mult = max(diff_mult, set_mult)
        
        # Track stats
        stats[difficulty] += 1
        if has_set_op:
            stats["set_ops"] += 1
        
        # Duplicate sample
        for _ in range(final_mult):
            oversampled.append(sample)
    
    print(f"Difficulty: easy={stats['easy']}, hard={stats['hard']}, extra={stats['extra']}")
    print(f"SET operations: {stats['set_ops']}")
    print(f"Original: {len(samples)} -> Oversampled: {len(oversampled)}")
    
    return oversampled


def main(args):
    print(f"Starting Preprocessing Pipeline")
    print(f"Model: {args.model}")
    print(f"Schema Format: {args.schema_format}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    print()

    # Select template based on model
    if args.model == "qwen":
        template = Qwen25Template()
        model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    elif args.model == "phi3":
        template = Phi3Template()
        model_id = "microsoft/Phi-3-mini-4k-instruct"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Using template: {template.name}")
    print(f"Model ID: {model_id}")

    # --- Load Schemas ---
    print("\nLoading schemas...")
    if not TABLES_JSON.exists():
        raise FileNotFoundError(f"Missing {TABLES_JSON}")

    with open(TABLES_JSON, "r") as f:
        tables_data = json.load(f)
    db_schemas = {db["db_id"]: db for db in tables_data}
    print(f"Loaded {len(db_schemas)} database schemas")

    # --- Load Datasets ---
    print("\nLoading datasets...")
    with open(TRAIN_SPIDER_JSON, "r") as f:
        train_spider = json.load(f)
    with open(TRAIN_OTHERS_JSON, "r") as f:
        train_others = json.load(f)
    with open(DEV_JSON, "r") as f:
        raw_dev = json.load(f)

    raw_train = train_spider + train_others
    print(f"Train: {len(raw_train)} samples ({len(train_spider)} spider + {len(train_others)} others)")
    print(f"Dev: {len(raw_dev)} samples")

    # --- Check for Data Leakage ---
    print("\nChecking for data leakage...")
    train_db_ids = {ex["db_id"] for ex in raw_train}
    dev_db_ids = {ex["db_id"] for ex in raw_dev}

    if not train_db_ids.isdisjoint(dev_db_ids):
        overlap = train_db_ids.intersection(dev_db_ids)
        raise ValueError(f"CRITICAL: DB Leakage detected! Overlap: {overlap}")
    print("No leakage detected. Train/Dev schemas are disjoint.")

    # --- Oversampling (Fix #2 & #3) ---
    if args.oversample:
        print("\nApplying combined oversampling (max of difficulty & SET op multipliers)...")
        raw_train = oversample_combined(
            raw_train,
            difficulty_multipliers={"easy": 1, "hard": 2, "extra": 3},
            set_op_multiplier=4
        )

    # --- Format Data ---
    print("\nFormatting training data...")
    if args.case_augment:
        print("  Case augmentation ENABLED (Run #3 fix)")
    train_formatted = [
        format_sample(ex, db_schemas, template, args.schema_format, case_augment=args.case_augment)
        for ex in tqdm(raw_train, desc="Train")
    ]

    print("Formatting validation data...")
    # Note: Don't augment validation data - we want to evaluate on original casing
    dev_formatted = [
        format_sample(ex, db_schemas, template, args.schema_format, case_augment=False)
        for ex in tqdm(raw_dev, desc="Dev")
    ]

    train_dataset = Dataset.from_list(train_formatted)
    eval_dataset = Dataset.from_list(dev_formatted)

    # --- Tokenizer Check & Filtering ---
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Using fallback tokenizer for length estimation...")
        tokenizer = None

    if tokenizer and args.filter_long:
        print(f"  Filtering sequences > {args.max_seq_length} tokens...")
        
        def is_valid_length(sample):
            tokens = tokenizer(sample["text"], truncation=False)["input_ids"]
            return len(tokens) <= args.max_seq_length

        original_len = len(train_dataset)
        train_dataset = train_dataset.filter(is_valid_length)
        filtered_count = original_len - len(train_dataset)
        print(f"Filtered {filtered_count} samples exceeding {args.max_seq_length} tokens.")

    # --- Save Datasets ---
    output_path = OUTPUT_DIR / args.model
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train"
    eval_path = output_path / "validation"

    print(f"\nSaving datasets to {output_path}...")
    train_dataset.save_to_disk(str(train_path))
    eval_dataset.save_to_disk(str(eval_path))

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUCCESS! Preprocessing complete.")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Template: {template.name} (v{template.version})")
    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(eval_dataset)}")
    print(f"Output: {output_path}")
    print()

    # Preview
    print("--- SAMPLE PREVIEW ---")
    print(train_dataset[0]["text"][:1000])
    print("...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Spider dataset for Text-to-SQL")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "phi3"],
        help="Model to prepare data for (qwen or phi3)"
    )
    parser.add_argument(
        "--schema-format",
        type=str,
        default="structured",
        choices=["structured", "ddl"],
        help="Schema serialization format"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens"
    )
    parser.add_argument(
        "--filter-long",
        action="store_true",
        default=True,
        help="Filter samples exceeding max sequence length"
    )
    parser.add_argument(
        "--no-filter",
        action="store_false",
        dest="filter_long",
        help="Don't filter long samples"
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        default=True,
        help="Apply difficulty + SET operation oversampling (Run #2 fix)"
    )
    parser.add_argument(
        "--no-oversample",
        action="store_false",
        dest="oversample",
        help="Disable oversampling"
    )
    parser.add_argument(
        "--case-augment",
        action="store_true",
        default=False,
        help="Apply schema case augmentation (Run #3 fix: forces model to read schema)"
    )
    
    args = parser.parse_args()
    main(args)

