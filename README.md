# Text-to-SQL with Fine-Tuned LLMs

> Convert natural language questions into SQL queries using fine-tuned language models.

## Overview

This project fine-tunes small language models using **QLoRA** (Quantized Low-Rank Adaptation) on the [Spider dataset](https://yale-lily.github.io/spider) to specialize in text-to-SQL conversion. The fine-tuned models can generate syntactically correct and semantically accurate SQL queries from natural language questions, given a database schema.

### Key Features

- **Fine-tuned Models**: Phi-3-mini and Qwen2.5-Coder trained on Spider dataset
- **FastAPI Service**: REST API for SQL generation
- **Interactive Demo**: Streamlit UI for testing queries
- **Efficient Training**: QLoRA enables training 7B models on consumer GPUs

## Results

### Execution Accuracy on Spider Dev Set

| Model | Base | Fine-Tuned | Improvement |
|-------|------|------------|-------------|
| **Qwen2.5-Coder-7B** | 73.3% | **74.1%** | +0.8% |
| **Phi-3-mini-4k** | 30.7% | **51.0%** | +20.3% |

### Training Impact

| Metric | Before | After |
|--------|--------|-------|
| Chatty Output (markdown/explanations) | 96.1% | **0%** |
| Exact Match Accuracy | 24.6% | **45.0%** |

> Fine-tuning improved output format compliance and exact match accuracy by a high margin.


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-to-sql.git
cd text-to-sql

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Run the API

```bash
# Start FastAPI server
uvicorn src.inference.api:app --reload --port 8000
```

### Run the Demo

```bash
# In a separate terminal
streamlit run src/inference/app.py
```

### API Usage

```bash
curl -X POST http://localhost:8000/generate_sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "Table: users\nColumns: id, name, age",
    "question": "How many users are older than 30?"
  }'
```

Response:
```json
{
  "sql": "SELECT COUNT(*) FROM users WHERE age > 30",
  "execution_time_ms": 245.5,
  "model": "phi3-finetuned"
}
```

## Project Structure

```
text-to-sql/
├── src/
│   ├── data/
│   │   ├── preprocess.py      # Data preprocessing pipeline
│   │   └── prompts/           # Prompt templates
│   ├── training/
│   │   └── train.py           # Training script (Colab)
│   ├── evaluation/
│   │   └── evaluate.py        # Evaluation metrics
│   └── inference/
│       ├── api.py             # FastAPI service
│       └── app.py             # Streamlit demo
├── checkpoints/               # Model adapters
├── results/                   # Evaluation results
└── spider_data/               # Spider dataset
```

## Training Details

### Training Data

- **Dataset**: Spider (Yale)
- **Training Samples**: ~7,000 (with oversampling)
- **Oversampling Strategy**: Hard queries 2x, Extra-hard 3x, SET operations 4x

## Technologies

- **LLM Framework**: Transformers, PEFT, Unsloth
- **Quantization**: bitsandbytes (4-bit)
- **Training**: QLoRA on Google Colab (T4)
- **API**: FastAPI, Uvicorn
- **Demo**: Streamlit
- **Database**: SQLite

