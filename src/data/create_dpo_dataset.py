"""
Create DPO Dataset from Training Predictions
=============================================
Processes raw predictions and creates DPO pairs:
- Wrong predictions → DPO(Gold, Hallucination)
- Correct predictions → DPO(Gold, Case-Augmented Synthetic)

Usage:
    python -m src.data.create_dpo_dataset \
        --predictions predictions/training_predictions_qwen.json \
        --output processed_data/dpo_qwen
"""

import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
from tqdm import tqdm


def augment_case(sql: str) -> str:
    """
    Apply case augmentation to create synthetic negative.
    Changes identifiers to different casing to create a "wrong" version.
    """
    # Common identifier patterns to modify
    patterns = [
        (r'\bname\b', 'NAME'),
        (r'\bNAME\b', 'name'),
        (r'\bid\b', 'ID'),
        (r'\bID\b', 'id'),
        (r'\bemail\b', 'EMAIL'),
        (r'\bEMAIL\b', 'email'),
        (r'\bage\b', 'AGE'),
        (r'\bAGE\b', 'age'),
        (r'\bcount\b', 'COUNT'),
        (r'\bCOUNT\b', 'count'),
    ]
    
    modified_sql = sql
    
    # Try each pattern until one matches
    for pattern, replacement in patterns:
        if re.search(pattern, sql):
            modified_sql = re.sub(pattern, replacement, sql, count=1)
            if modified_sql != sql:
                return modified_sql
    
    # Fallback: Add DISTINCT if not present
    if "DISTINCT" not in sql.upper() and sql.upper().startswith("SELECT"):
        modified_sql = sql.replace("SELECT", "SELECT DISTINCT", 1)
        return modified_sql
    
    # If nothing works, swap first identifier's case
    words = sql.split()
    for i, word in enumerate(words):
        if word.isidentifier() and word.lower() not in ['select', 'from', 'where', 'and', 'or', 'join', 'on', 'group', 'by', 'order', 'limit', 'as']:
            if word.isupper():
                words[i] = word.lower()
            else:
                words[i] = word.upper()
            return " ".join(words)
    
    return None  # Could not create synthetic


def create_dpo_dataset(predictions_path: str, synthetic_ratio: float = 1.0) -> List[Dict]:
    """
    Create DPO dataset from predictions.
    
    Args:
        predictions_path: Path to training_predictions_*.json
        synthetic_ratio: Ratio of correct samples to use for synthetic pairs (0-1)
    
    Returns:
        List of DPO samples
    """
    print(f"Loading predictions from: {predictions_path}")
    with open(predictions_path) as f:
        predictions = json.load(f)
    
    stats = {
        "total": len(predictions),
        "correct": 0,
        "incorrect": 0,
        "dpo_real": 0,
        "dpo_synthetic": 0,
        "skipped": 0,
    }
    
    dpo_samples = []
    
    for pred in tqdm(predictions, desc="Creating DPO pairs"):
        prompt = pred["prompt"]
        gold_sql = pred["gold_sql"]
        predicted_sql = pred["predicted_sql"]
        is_correct = pred["is_correct"]
        
        if is_correct:
            stats["correct"] += 1
            
            # Randomly decide whether to include (based on ratio)
            if random.random() > synthetic_ratio:
                stats["skipped"] += 1
                continue
            
            # Create synthetic negative via case augmentation
            synthetic_rejected = augment_case(gold_sql)
            
            if synthetic_rejected and synthetic_rejected != gold_sql:
                dpo_samples.append({
                    "prompt": prompt,
                    "chosen": gold_sql,
                    "rejected": synthetic_rejected,
                    "type": "synthetic",
                    "db_id": pred.get("db_id", "unknown"),
                })
                stats["dpo_synthetic"] += 1
            else:
                stats["skipped"] += 1
        else:
            stats["incorrect"] += 1
            
            # Use actual hallucination as rejected
            if predicted_sql and len(predicted_sql.strip()) > 3:
                dpo_samples.append({
                    "prompt": prompt,
                    "chosen": gold_sql,
                    "rejected": predicted_sql,
                    "type": "real",
                    "db_id": pred.get("db_id", "unknown"),
                })
                stats["dpo_real"] += 1
            else:
                stats["skipped"] += 1
    
    print("\n" + "=" * 60)
    print("DPO DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"Total predictions: {stats['total']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Incorrect: {stats['incorrect']}")
    print(f"\nDPO Pairs Created: {len(dpo_samples)}")
    print(f"  Real (from hallucinations): {stats['dpo_real']}")
    print(f"  Synthetic (case augmented): {stats['dpo_synthetic']}")
    print(f"  Skipped: {stats['skipped']}")
    
    return dpo_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to training_predictions_*.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed_data/dpo",
        help="Output directory"
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.5,
        help="Ratio of correct predictions to use for synthetic pairs (0-1)"
    )
    
    args = parser.parse_args()
    
    # Create DPO dataset
    dpo_samples = create_dpo_dataset(
        predictions_path=args.predictions,
        synthetic_ratio=args.synthetic_ratio,
    )
    
    if not dpo_samples:
        print("No DPO samples created!")
        return
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "prompt": [s["prompt"] for s in dpo_samples],
        "chosen": [s["chosen"] for s in dpo_samples],
        "rejected": [s["rejected"] for s in dpo_samples],
    })
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_disk(str(output_path / "train"))
    print(f"\nDataset saved to: {output_path / 'train'}")
    
    # Also save JSON for inspection
    json_path = output_path / "dpo_samples.json"
    with open(json_path, "w") as f:
        json.dump(dpo_samples, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE DPO PAIRS")
    print("=" * 60)
    
    real_samples = [s for s in dpo_samples if s["type"] == "real"][:2]
    synthetic_samples = [s for s in dpo_samples if s["type"] == "synthetic"][:2]
    
    print("\n--- REAL (from hallucinations) ---")
    for s in real_samples:
        print(f"  Chosen:   {s['chosen'][:50]}...")
        print(f"  Rejected: {s['rejected'][:50]}...")
    
    print("\n--- SYNTHETIC (case augmented) ---")
    for s in synthetic_samples:
        print(f"  Chosen:   {s['chosen'][:50]}...")
        print(f"  Rejected: {s['rejected'][:50]}...")


if __name__ == "__main__":
    main()
