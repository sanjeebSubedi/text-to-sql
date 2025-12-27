import json
import os

import sqlparse
from datasets import Dataset
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "spider_data"
OUTPUT_DIR = "spider_processed_v3"
MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# Global Cache to speed up processing
SCHEMA_CACHE = {}


# ==========================================
# 1. DETERMINISTIC SCHEMA SERIALIZER (Cached)
# ==========================================
def serialize_schema_sorted(db_id, db_schemas):
    """
    Serializes schema with strict priority:
    1. Primary Keys (Group 0)
    2. Foreign Keys (Group 1)
    3. Remaining columns (Group 2)
    """
    # 🔧 Improvement 3: Schema Caching
    if db_id in SCHEMA_CACHE:
        return SCHEMA_CACHE[db_id]

    if db_id not in db_schemas:
        return ""

    schema_info = db_schemas[db_id]

    # --- Unpack Metadata ---
    table_names = schema_info["table_names_original"]
    column_names = schema_info["column_names_original"]  # [[table_idx, name], ...]
    column_types = schema_info["column_types"]
    primary_keys = set(schema_info["primary_keys"])
    foreign_keys = schema_info["foreign_keys"]  # [[col_idx, ref_col_idx], ...]

    # --- Build FK Map ---
    fk_map = {}
    for col_idx, ref_col_idx in foreign_keys:
        ref_table_idx, ref_col_name = column_names[ref_col_idx]
        ref_table_name = table_names[ref_table_idx]
        fk_map[col_idx] = f"foreign key -> {ref_table_name}.{ref_col_name}"

    # --- Build Table Objects ---
    tables = {i: [] for i in range(len(table_names))}

    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue  # Skip '*' wildcard

        # Build Description
        details = [column_types[idx]]
        is_pk = idx in primary_keys
        is_fk = idx in fk_map

        if is_pk:
            details.append("primary key")
        if is_fk:
            details.append(fk_map[idx])

        col_str = f"{col_name} ({', '.join(details)})"

        # Store with sorting metadata
        tables[table_idx].append(
            {"text": col_str, "is_pk": is_pk, "is_fk": is_fk, "orig_idx": idx}
        )

    # --- Format Output with Strict Sorting ---
    schema_lines = []
    for i, table_name in enumerate(table_names):
        cols = tables[i]

        # ✅ Fix 1: Explicit Priority Logic
        def sort_key(c):
            if c["is_pk"]:
                return (0, c["orig_idx"])  # Priority 1: PK
            if c["is_fk"]:
                return (1, c["orig_idx"])  # Priority 2: FK
            return (2, c["orig_idx"])  # Priority 3: Others

        sorted_cols = sorted(cols, key=sort_key)

        col_strings = [c["text"] for c in sorted_cols]
        cols_formatted = ", ".join(col_strings)
        schema_lines.append(f"Table: {table_name}\nColumns: {cols_formatted}")

    result = "\n\n".join(schema_lines)
    SCHEMA_CACHE[db_id] = result  # Cache it
    return result


# ==========================================
# 2. SAFE SQL NORMALIZER
# ==========================================
def normalize_sql(query):
    """
    Attempts to uppercase keywords using sqlparse.
    Fallbacks to raw query on failure to prevent silent data corruption.
    """
    # ✅ Fix 2: Safety Wrapper
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
        print(f"⚠️ SQL Normalization failed for: {query[:50]}... Error: {e}")
        return query.strip()


# ==========================================
# 3. PROMPT FORMATTER
# ==========================================
def format_instruction(sample, db_schemas):
    db_id = sample["db_id"]
    question = sample["question"]
    raw_query = sample["query"]

    schema_context = serialize_schema_sorted(db_id, db_schemas)
    target_sql = normalize_sql(raw_query)

    sys_prompt = (
        "You are a text-to-SQL AI assistant. "
        "Your goal is to output valid, executable SQL for the given SQLite schema. "
        "Pay attention to primary and foreign keys."
    )

    # Llama-3 Chat Format
    formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Database Schema:
{schema_context}

### Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_sql}<|eot_id|>"""

    return {
        "text": formatted_text,
        "db_id": db_id,  # 🔧 Improvement 2: Meta-data for analysis
        "raw_query": raw_query,
        "target_sql": target_sql,
    }


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(f"🚀 Starting Preprocessing Pipeline (v3.0)...")

    # --- A. Load Schemas ---
    tables_path = os.path.join(DATA_DIR, "tables.json")
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"Missing {tables_path}")

    with open(tables_path, "r") as f:
        tables_data = json.load(f)
    db_schemas = {db["db_id"]: db for db in tables_data}

    # --- B. Load & Merge Datasets ---
    with open(os.path.join(DATA_DIR, "train_spider.json"), "r") as f:
        train_spider = json.load(f)
    with open(os.path.join(DATA_DIR, "train_others.json"), "r") as f:
        train_others = json.load(f)

    raw_train = train_spider + train_others

    with open(os.path.join(DATA_DIR, "dev.json"), "r") as f:
        raw_dev = json.load(f)

    # --- C. ❌ Fix 3: Strict Leakage Check ---
    print("🔎 Checking for Data Leakage...")
    train_db_ids = {ex["db_id"] for ex in raw_train}
    dev_db_ids = {ex["db_id"] for ex in raw_dev}

    if not train_db_ids.isdisjoint(dev_db_ids):
        overlap = train_db_ids.intersection(dev_db_ids)
        raise ValueError(f"❌ CRITICAL: DB Leakage detected! Overlap: {overlap}")
    print("✅ No Leakage detected. Train/Dev schemas are disjoint.")

    # --- D. Formatting ---
    print("⚙️ Formatting training data...")
    train_data_formatted = [format_instruction(ex, db_schemas) for ex in raw_train]

    print("⚙️ Formatting validation data...")
    dev_data_formatted = [format_instruction(ex, db_schemas) for ex in raw_dev]

    train_dataset = Dataset.from_list(train_data_formatted)
    eval_dataset = Dataset.from_list(dev_data_formatted)

    # --- E. Tokenizer Checks & Filtering ---
    print("📚 Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Warning: Could not load specific tokenizer ({e}). Using base Llama-3.")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # ✅ Fix 4: Verify Special Tokens
    required_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    vocab = tokenizer.get_vocab()
    missing_tokens = [t for t in required_tokens if t not in vocab]

    if missing_tokens:
        raise ValueError(
            f"❌ Tokenizer is missing critical Llama-3 tokens: {missing_tokens}"
        )
    print("✅ Tokenizer vocabulary verified.")

    print("✂️ Filtering long sequences...")

    def is_valid_length(sample):
        return len(tokenizer(sample["text"])["input_ids"]) <= MAX_SEQ_LENGTH

    original_len = len(train_dataset)
    train_dataset = train_dataset.filter(is_valid_length)
    print(
        f"Filtered {original_len - len(train_dataset)} samples exceeded {MAX_SEQ_LENGTH} tokens."
    )

    # --- F. Save ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
    eval_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "validation"))

    print(f"\n✅ SUCCESS! Processed data saved to {OUTPUT_DIR}")
    print(f"   Train Size: {len(train_dataset)}")
    print(f"   Val Size:   {len(eval_dataset)}")

    # Final Visual Check
    print("\n--- SAMPLE INPUT PREVIEW (Check Sorting) ---")
    print(train_dataset[0]["text"][:800])


if __name__ == "__main__":
    main()
