# Text-to-SQL Fine-Tuning Implementation Plan

> **Project Goal**: Fine-tune a small LLM on the Spider dataset to specialize in converting natural language questions into SQL queries.

> **Target Outcome**: A portfolio-ready project demonstrating AI engineering skills for LLM/AI-related job applications.

---

## Project Overview

This project will fine-tune a small language model using parameter-efficient techniques (LoRA/QLoRA) on the Spider dataset from Yale. The model will learn to generate syntactically correct and semantically accurate SQL queries from natural language questions, given a database schema context.

### Base Model Candidates (Finalized)

| Model | Size | Rationale |
|-------|------|-----------|
| **Qwen2.5-Coder-7B-Instruct** | 7B | Primary choice - state-of-the-art code generation, strong SQL understanding |
| **Phi-3-mini-4k-instruct** | 3.8B | Fallback - smaller footprint for faster iteration and lower VRAM usage |

> [!NOTE]
> Training will be performed on **Google Colab** (T4/A100 GPU). Local RTX 3050 available for testing and inference.

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080/4070 or better recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space (for dataset, checkpoints, and model weights)

---

## Phase 1: Setup & Data Engineering

### 1.1 Environment Setup

- [ ] Python 3.10+ environment configuration
- [ ] Install core dependencies:
  - PyTorch (CUDA-enabled)
  - Transformers
  - PEFT (Parameter-Efficient Fine-Tuning)
  - bitsandbytes (4-bit quantization)
  - datasets (HuggingFace)
  - accelerate
- [ ] Install evaluation dependencies:
  - sqlparse
  - sqlite3
- [ ] Install experiment tracking:
  - **Weights & Biases (wandb)** or MLflow
- [ ] Configure CUDA and verify GPU availability
- [ ] Set global random seeds for reproducibility (seed=42)

### 1.2 Dataset Acquisition

- [ ] Download Spider train/dev splits from Yale
- [ ] Download all SQLite database files (140+ databases)
- [ ] Verify schema-disjoint property between train/dev splits
- [ ] Organize dataset in `spider_data/` directory

### 1.3 Dataset Exploration

- [ ] Inspect Spider JSON structure:
  - `db_id`, `table_names_original`
  - `column_names_original`
  - Foreign key mappings
- [ ] Compute dataset statistics:
  - Number of databases, tables, columns
  - Average schema size (tables/columns per database)
  - SQL query length distribution
  - JOIN frequency analysis
  - Query complexity distribution (Spider difficulty levels)
- [ ] Document findings in `spider_dataset_knowledge_base.md`

### 1.4 SQL Normalization

- [ ] Canonicalize SQL formatting (uppercase keywords, consistent spacing)
- [ ] Normalize alias usage (T1, T2, etc.)
- [ ] Validate all SQL queries execute successfully on corresponding databases
- [ ] Log and investigate any failed queries

### 1.5 Prompt Template Design

Define instruction format with versioning:

```text
[PROMPT TEMPLATE v1.0]

### System:
You are an expert SQL assistant. Given a database schema and a natural language question, 
generate the correct SQL query. Output only the SQL query without any explanation.

### Schema:
{schema_ddl}

### Question:
{question}

### SQL:
{sql_query}
```

- [ ] Design prompt template with clear sections
- [ ] Support multiple schema representation formats (DDL-style, structured text)
- [ ] Track prompt versions in `prompts/` directory
- [ ] Document template design decisions

### 1.6 Data Preprocessing Pipeline

- [ ] Parse Spider JSON files (`train_spider.json`, `dev.json`)
- [ ] Serialize database schemas (DDL-style or structured text)
- [ ] Insert schema + question into prompt template
- [ ] Tokenize samples to detect context window overflow
- [ ] Handle problematic samples:
  - Log samples exceeding max context length
  - Apply truncation strategy or drop samples
- [ ] Create train/validation splits

### 1.7 Dataset Versioning & Artifacts

- [ ] Save processed dataset as immutable artifact (HuggingFace datasets format)
- [ ] Log comprehensive statistics:
  - Sample counts (train/dev)
  - Token length distribution
  - Schema complexity distribution
- [ ] Version datasets with clear naming convention

---

## Phase 2: Schema Pruning & Linking (Optional Enhancement)

> [!TIP]
> This phase is optional but highly recommended. It addresses the context window limitation and demonstrates advanced retrieval techniques. Can be implemented as a "v2" enhancement if time-constrained.

### 2.1 Schema Embedding

- [ ] Select sentence embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- [ ] Embed table names and column names for each database
- [ ] Store embeddings per database in FAISS index or similar

### 2.2 Question-to-Schema Retrieval

- [ ] Embed user questions using same embedding model
- [ ] Retrieve top-k relevant tables/columns via cosine similarity
- [ ] Tune k value based on recall of ground-truth tables

### 2.3 Schema Pruning Strategy

- [ ] Construct reduced schema containing only retrieved tables
- [ ] Preserve foreign keys among selected tables
- [ ] Validate pruned schemas maintain query executability

### 2.4 Training Data Augmentation

- [ ] Generate two dataset variants:
  - **Full schema**: Complete database schema
  - **Pruned schema**: Only relevant tables/columns
- [ ] Tag samples with schema mode for ablation studies

### 2.5 Ablation Experiments (Post-Training)

- [ ] Train model with full schema only
- [ ] Train model with pruned schema only
- [ ] Compare execution accuracy between variants
- [ ] Document findings and trade-offs

---

## Phase 3: Fine-Tuning

### 3.1 Baseline Evaluation (Before Training)

- [ ] Run zero-shot inference using base model(s)
- [ ] Test on sample of dev set (50-100 examples)
- [ ] Record baseline metrics:
  - Execution accuracy
  - Exact match accuracy
  - Syntax error rate
- [ ] Document baseline results for comparison

### 3.2 Model Initialization

- [ ] Load base model in 4-bit quantization (QLoRA)
- [ ] Freeze base model weights
- [ ] Verify model loads correctly on available hardware
- [ ] Document VRAM usage

### 3.3 LoRA Configuration

```python
# Recommended starting configuration
lora_config = LoraConfig(
    r=16,                    # Rank (start with 16, can tune later)
    lora_alpha=32,           # Alpha = 2x rank is common
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"       # MLP layers (optional)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

- [ ] Configure LoRA with justified hyperparameters
- [ ] Document target module selection rationale
- [ ] Calculate trainable parameter count

### 3.4 Training Configuration

- [ ] Define training hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| LR Schedule | Cosine | Smooth decay |
| Batch Size | 4-8 | Adjust based on VRAM |
| Gradient Accumulation | 4-8 | Effective batch size = 16-32 |
| Max Sequence Length | 2048-4096 | Based on schema sizes |
| Epochs | 2-3 | Avoid overfitting |
| Warmup Ratio | 0.03 | ~3% of total steps |

- [ ] Log all hyperparameters to wandb/MLflow
- [ ] Configure mixed precision training (fp16/bf16)

### 3.5 Training Execution

- [ ] Run training for 2-3 epochs
- [ ] Monitor training loss in real-time
- [ ] Periodic validation inference (every N steps):
  - Not just validation loss
  - Actual SQL generation on held-out samples
  - Quick execution accuracy check

### 3.6 Checkpointing Strategy

- [ ] Save LoRA adapters per epoch
- [ ] Save best checkpoint based on validation execution accuracy
- [ ] Implement early stopping if validation performance degrades
- [ ] Store checkpoints in `checkpoints/` directory

### 3.7 Training Diagnostics

- [ ] Plot and analyze:
  - Training loss curve
  - Validation loss curve
  - Validation execution accuracy trend
  - Learning rate schedule
- [ ] Detect and address:
  - Overfitting (train loss ↓, val loss ↑)
  - Underfitting (both losses plateau high)
  - Instability (loss spikes)

---

## Phase 4: Evaluation & Error Analysis

### 4.1 Inference Pipeline

- [ ] Implement deterministic decoding:
  - Temperature = 0 or greedy decoding
  - No sampling during evaluation
- [ ] SQL-only output enforcement:
  - Post-processing to extract SQL from generation
  - Handle edge cases (extra text, explanations)

### 4.2 Exact Match Evaluation

- [ ] Normalize predicted and ground-truth SQL:
  - Lowercase
  - Remove extra whitespace
  - Standardize quotes
- [ ] Compute exact match accuracy
- [ ] Note: This is a secondary metric (too strict)

### 4.3 Execution Accuracy (Primary Metric)

- [ ] Execute predicted SQL on corresponding SQLite database
- [ ] Execute ground-truth SQL on same database
- [ ] Compare result sets:
  - Order-insensitive comparison
  - Handle NULL values correctly
- [ ] Compute execution accuracy percentage
- [ ] Report accuracy by Spider difficulty level:
  - Easy
  - Medium
  - Hard
  - Extra Hard

### 4.4 Error Categorization

Implement automatic error classification:

| Error Type | Description | Detection Method |
|------------|-------------|------------------|
| Syntax Error | Invalid SQL syntax | SQLite parse error |
| Missing JOIN | Required join not present | Missing expected tables in FROM |
| Wrong Aggregation | COUNT vs SUM, etc. | Result value mismatch |
| Hallucinated Column | Column doesn't exist | SQLite column not found error |
| Wrong Filter | Incorrect WHERE condition | Result set differs |
| Wrong Table | Correct columns, wrong table | Parse SQL and compare |

- [ ] Implement error classification logic
- [ ] Log error type for each failed prediction

### 4.5 Quantitative Breakdown

- [ ] Generate error distribution chart (pie/bar chart)
- [ ] Compute accuracy breakdown:
  - By difficulty level
  - By number of tables involved
  - By presence of aggregations
  - By presence of subqueries
- [ ] Compare against baseline (zero-shot)
- [ ] Calculate improvement percentage

### 4.6 Qualitative Case Studies

- [ ] Select 5-10 representative failure cases
- [ ] For each failure, document:
  - Input question
  - Database schema
  - Ground-truth SQL
  - Predicted SQL
  - Error category
  - **Root cause analysis**: Why did the model fail?
- [ ] Identify patterns in failures
- [ ] Suggest potential improvements

---

## Phase 5: Deployment & Demo

### 5.1 Inference Service

- [ ] Build FastAPI service with endpoints:

```python
POST /generate_sql
{
    "schema": "CREATE TABLE ...",  # or schema dict
    "question": "How many students are enrolled?"
}

Response:
{
    "sql": "SELECT COUNT(*) FROM students",
    "confidence": 0.95,  # optional
    "execution_time_ms": 150
}
```

- [ ] Implement proper error handling
- [ ] Add request validation

### 5.2 SQL Validation Layer

- [ ] Parse generated SQL before execution
- [ ] Validate table/column existence against provided schema
- [ ] Reject potentially unsafe queries:
  - DROP, DELETE, UPDATE, INSERT
  - Only allow SELECT statements
- [ ] Return meaningful error messages

### 5.3 Demo Application (Streamlit)

- [ ] Build interactive UI with:
  - Database/schema selector dropdown
  - Natural language query input
  - Generated SQL display with syntax highlighting
  - "Execute" button to run SQL
  - Results table display
- [ ] Include sample questions for demo
- [ ] Add option to view database schema

### 5.4 Performance Profiling

- [ ] Measure and document:
  - Inference latency (ms per request)
  - VRAM usage during inference
  - Tokens per second throughput
  - Cold start time
- [ ] Optimize if needed (batching, caching)

### 5.5 Containerization (Optional)

- [ ] Create Dockerfile for reproducible deployment
- [ ] Support both GPU and CPU inference
- [ ] Include docker-compose for easy setup
- [ ] Document deployment instructions

---

## Phase 6: Documentation & Resume Packaging

### 6.1 Repository Structure

```
text_to_sql/
├── README.md                    # Project overview, results, usage
├── implementation_plan.md       # This document
├── spider_dataset_knowledge_base.md
├── requirements.txt             # or pyproject.toml
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Data preprocessing
│   │   ├── schema_linker.py     # Schema pruning (Phase 2)
│   │   └── prompts/             # Prompt templates
│   ├── training/
│   │   ├── train.py             # Training script
│   │   └── config.py            # Hyperparameters
│   ├── evaluation/
│   │   ├── evaluate.py          # Evaluation script
│   │   └── error_analysis.py    # Error categorization
│   └── inference/
│       ├── api.py               # FastAPI service
│       └── app.py               # Streamlit demo
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── checkpoints/                 # Saved model adapters
├── results/                     # Evaluation results, charts
└── docker/
    └── Dockerfile
```

### 6.2 README Content

- [ ] Project title and one-line description
- [ ] Architecture diagram (visual overview)
- [ ] Dataset details (Spider, statistics)
- [ ] Model choice rationale
- [ ] Training details (LoRA config, epochs, etc.)
- [ ] **Evaluation results table**:

| Model | Exec Acc (Easy) | Exec Acc (Medium) | Exec Acc (Hard) | Exec Acc (Extra) | Overall |
|-------|-----------------|-------------------|-----------------|------------------|---------|
| Base (zero-shot) | X% | X% | X% | X% | X% |
| Fine-tuned | X% | X% | X% | X% | X% |

- [ ] Limitations and future work
- [ ] Setup and usage instructions
- [ ] Demo screenshots/GIF

### 6.3 Resume Bullets (Draft)

Prepare polished resume bullets:

1. **Fine-tuned a 7B parameter LLM** using QLoRA on the Spider dataset, achieving **X% execution accuracy** on text-to-SQL conversion (Y% improvement over baseline)

2. **Designed and implemented schema pruning** using sentence embeddings and FAISS, reducing input context by Z% while maintaining accuracy

3. **Built end-to-end ML pipeline** including data preprocessing, training, evaluation with error analysis, and deployment via FastAPI + Streamlit

4. **Technologies**: PyTorch, Transformers, PEFT, HuggingFace, FastAPI, Streamlit, Docker

### 6.4 Interview Talking Points

Prepare answers to anticipated questions:

| Question | Key Points |
|----------|------------|
| **Why LoRA/QLoRA?** | Memory efficient, trains only ~0.1% of parameters, enables fine-tuning 7B models on consumer GPUs |
| **Why execution accuracy over exact match?** | Multiple valid SQL formulations, exact match too strict, execution reflects real-world utility |
| **Why schema pruning?** | Context window limitations, reduces noise, mirrors production systems |
| **What were the main challenges?** | Schema representation, handling complex JOINs, error categorization |
| **How would you improve this?** | Better schema linking, self-consistency decoding, iterative refinement |

---

## Fallback Plans

> [!CAUTION]
> If fine-tuning doesn't yield satisfactory results, consider these alternatives:

1. **Few-shot prompting**: Use in-context examples instead of fine-tuning
2. **RAG with SQL examples**: Retrieve similar question-SQL pairs as context
3. **Smaller model**: Try Phi-2/Phi-3 if 7B is too resource-intensive
4. **Hybrid approach**: Combine fine-tuned model with retrieval

---

## Timeline Estimate

| Phase | Estimated Duration |
|-------|-------------------|
| Phase 1: Setup & Data Engineering | 3-4 days |
| Phase 2: Schema Linking (Optional) | 2-3 days |
| Phase 3: Fine-Tuning | 2-3 days |
| Phase 4: Evaluation & Analysis | 2-3 days |
| Phase 5: Deployment | 2-3 days |
| Phase 6: Documentation | 1-2 days |
| **Total** | **12-18 days** |

---

## Success Criteria

- [ ] Execution accuracy ≥ 60% on Spider dev set (baseline is typically ~10-20% zero-shot)
- [ ] Demo application functional and visually polished
- [ ] Clean, well-documented GitHub repository
- [ ] Ready for resume and interview discussions

---

*Document Version: 1.0*  
*Last Updated: 2025-12-29*
