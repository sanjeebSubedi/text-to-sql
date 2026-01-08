from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class Config:
    ADAPTER_PATH = "checkpoints/final_adapter_phi3"
    MODEL_TYPE = "phi3"  # "qwen" or "phi3"
    
    BASE_MODELS = {
        "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
    }
    
    MAX_NEW_TOKENS = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_IN_4BIT = True


SYSTEM_PROMPT = """You are a SQL query generator. Your ONLY task is to convert natural language questions into SQL queries.

CRITICAL RULES:
1. Output ONLY the raw SQL query - nothing else.
2. Do NOT include explanations, comments, or markdown.
3. Do NOT wrap the query in code blocks.
4. Use the EXACT table and column names from the schema (preserve original casing).
5. Do NOT use DISTINCT unless explicitly required by the question.
6. Do NOT add column aliases unless necessary for clarity.
7. Use SQLite syntax."""


# ==========================================
# Request/Response Models
# ==========================================
class SQLRequest(BaseModel):
    schema_text: str = Field(..., alias="schema", description="Database schema")
    question: str = Field(..., description="Natural language question")
    
    class Config:
        populate_by_name = True


class SQLResponse(BaseModel):
    sql: str
    execution_time_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Global Model State
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    
    print(f"Loading {Config.MODEL_TYPE} from {Config.ADAPTER_PATH}")
    base_model_id = Config.BASE_MODELS[Config.MODEL_TYPE]
    
    bnb_config = None
    if Config.LOAD_IN_4BIT and Config.DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if Config.DEVICE == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
    )
    
    model = PeftModel.from_pretrained(model, Config.ADAPTER_PATH)
    model.eval()
    print("Model loaded!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global model, tokenizer
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Text-to-SQL API",
    description="Generate SQL queries from natural language",
    version="1.0.0",
    lifespan=lifespan,
)


# Prompt Formatting
# Token constants
QWEN_IM_START = "<" + "|im_start|" + ">"
QWEN_IM_END = "<" + "|im_end|" + ">"
PHI_SYSTEM = "<" + "|system|" + ">"
PHI_USER = "<" + "|user|" + ">"
PHI_ASSISTANT = "<" + "|assistant|" + ">"
PHI_END = "<" + "|end|" + ">"


def format_prompt(schema: str, question: str) -> str:
    user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
    
    if Config.MODEL_TYPE == "qwen":
        return f"{QWEN_IM_START}system\n{SYSTEM_PROMPT}{QWEN_IM_END}\n{QWEN_IM_START}user\n{user_content}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
    else:
        return f"{PHI_SYSTEM}\n{SYSTEM_PROMPT} {PHI_END}\n{PHI_USER}\n{user_content} {PHI_END}\n{PHI_ASSISTANT}\n"


def extract_sql(response: str) -> str:
    """Extract SQL from model response."""
    if Config.MODEL_TYPE == "qwen":
        if "assistant" in response:
            parts = response.split(QWEN_IM_START + "assistant")
            if len(parts) > 1:
                return parts[-1].split(QWEN_IM_END)[0].strip()
    else:
        if PHI_ASSISTANT in response:
            return response.split(PHI_ASSISTANT)[-1].split(PHI_END)[0].strip()
    return response.strip()


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL is SELECT only (no dangerous operations)."""
    sql_upper = sql.upper().strip()
    
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return False, f"Dangerous SQL keyword detected: {keyword}"
    
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"
    
    return True, ""


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=Config.DEVICE,
    )


@app.post("/generate_sql", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Format prompt
    prompt = format_prompt(request.schema_text, request.question)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,  # Required for Phi-3 compatibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract SQL
    sql = extract_sql(response)
    
    # Validate
    is_valid, error_msg = validate_sql(sql)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    execution_time = (time.time() - start_time) * 1000
    
    return SQLResponse(
        sql=sql,
        execution_time_ms=round(execution_time, 2),
        model=f"{Config.MODEL_TYPE}-finetuned",
    )


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
