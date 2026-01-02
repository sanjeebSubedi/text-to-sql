import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "spider_data" / "spider_data"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "src" / "data" / "prompts"


for dir_path in [OUTPUT_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PROMPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

TABLES_JSON = DATA_DIR / "tables.json"
TRAIN_SPIDER_JSON = DATA_DIR / "train_spider.json"
TRAIN_OTHERS_JSON = DATA_DIR / "train_others.json"
DEV_JSON = DATA_DIR / "dev.json"
DATABASE_DIR = DATA_DIR / "database"


@dataclass
class ModelConfig:
    """Configuration for model selection and loading."""
    
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    fallback_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # or "bfloat16" for newer GPUs
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Model-specific settings
    trust_remote_code: bool = True
    use_flash_attention_2: bool = False  # Set True if GPU supports it
    
    # Token limits
    max_seq_length: int = 2048  # Adjust based on schema sizes
    
    def get_model_short_name(self) -> str:
        """Get a short name for the model for file naming."""
        return self.model_id.split("/")[-1].lower().replace("-", "_")


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    # LoRA hyperparameters
    r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha (typically 2x rank)
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Target modules for different models
    # For Qwen2.5-Coder:
    qwen_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # For Phi-3:
    phi3_target_modules: List[str] = field(default_factory=lambda: [
        "qkv_proj", "o_proj",
        "gate_up_proj", "down_proj"
    ])
    
    def get_target_modules(self, model_name: str) -> List[str]:
        """Get target modules based on model architecture."""
        if "qwen" in model_name.lower():
            return self.qwen_target_modules
        elif "phi" in model_name.lower():
            return self.phi3_target_modules
        else:
            # Default to common target modules
            return ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training params
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Effective batch size = per_device_batch_size * gradient_accumulation * num_gpus
    # With defaults: 4 * 4 * 1 = 16
    
    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Memory optimization
    fp16: bool = False  # Enable if GPU supports it
    bf16: bool = False  # Preferred if GPU supports it (Ampere+)
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Reproducibility
    seed: int = 42
    
    # Output
    output_dir: str = str(CHECKPOINTS_DIR)
    
    # Wandb
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.run_name is None:
            from datetime import datetime
            self.run_name = f"text2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    
    # Prompt template version
    prompt_version: str = "v1.0"
    
    # Schema serialization format
    schema_format: str = "structured"  # "ddl" or "structured"
    
    # Token limits
    max_seq_length: int = 2048
    
    # Filtering
    filter_long_sequences: bool = True
    max_sql_length: int = 500  # Characters
    
    # Train/val split (if needed)
    validation_split: float = 0.0  # Use official dev set
    
    # Difficulty filtering (optional)
    include_difficulties: Optional[List[str]] = None  # ["easy", "medium", "hard", "extra"]


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    
    # Decoding parameters
    temperature: float = 0.0  # Greedy decoding for evaluation
    top_p: float = 1.0
    top_k: int = 1
    max_new_tokens: int = 256
    do_sample: bool = False
    
    # Execution settings
    execution_timeout: int = 30  # Seconds per query
    
    # Metrics to compute
    compute_exact_match: bool = True
    compute_execution_accuracy: bool = True
    
    # Result saving
    save_predictions: bool = True
    save_error_analysis: bool = True



MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
EVAL_CONFIG = EvalConfig()


def setup_environment():
    """Set up environment variables and random seeds."""
    import random
    import numpy as np
    import torch
    
    seed = TRAINING_CONFIG.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"Environment set up with seed={seed}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print(" CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"\nModel: {MODEL_CONFIG.model_id}")
    print(f"Max Sequence Length: {MODEL_CONFIG.max_seq_length}")
    print(f"\nLoRA Config:")
    print(f"  Rank: {LORA_CONFIG.r}")
    print(f"  Alpha: {LORA_CONFIG.lora_alpha}")
    print(f"  Dropout: {LORA_CONFIG.lora_dropout}")
    print(f"\nTraining Config:")
    print(f"  Epochs: {TRAINING_CONFIG.num_train_epochs}")
    print(f"  Batch Size: {TRAINING_CONFIG.per_device_train_batch_size}")
    print(f"  Gradient Accumulation: {TRAINING_CONFIG.gradient_accumulation_steps}")
    print(f"  Learning Rate: {TRAINING_CONFIG.learning_rate}")
    print(f"  Scheduler: {TRAINING_CONFIG.lr_scheduler_type}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_environment()
    print_config()
