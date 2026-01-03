from dataclasses import dataclass
from typing import Optional, List

PROMPT_VERSION = "v2.0"

SYSTEM_PROMPT = """You are a SQL query generator. Your ONLY task is to convert natural language questions into SQL queries.

CRITICAL RULES:
1. Output ONLY the raw SQL query - nothing else.
2. Do NOT include explanations, comments, or markdown.
3. Do NOT wrap the query in code blocks.
4. Use the EXACT table and column names from the schema (preserve original casing).
5. Do NOT use DISTINCT unless explicitly required by the question.
6. Do NOT add column aliases unless necessary for clarity.
7. Use SQLite syntax.

Example:
Question: "How many employees are there?"
Output: SELECT COUNT(*) FROM employees"""



def get_qwen_tokens():
    """Return Qwen special tokens."""
    return {
        "im_start": chr(60) + "|im_start|" + chr(62),
        "im_end": chr(60) + "|im_end|" + chr(62),
    }


def get_phi_tokens():
    """Return Phi-3 special tokens."""
    return {
        "system": chr(60) + "|system|" + chr(62),
        "user": chr(60) + "|user|" + chr(62),
        "assistant": chr(60) + "|assistant|" + chr(62),
        "end": chr(60) + "|end|" + chr(62),
    }


@dataclass
class Qwen25Template:
    """Prompt template for Qwen2.5-Coder-Instruct (ChatML format)."""
    
    name: str = "qwen25_coder"
    version: str = PROMPT_VERSION
    system_prompt: str = SYSTEM_PROMPT
    
    def format_prompt(
        self,
        schema: str,
        question: str,
        sql: Optional[str] = None,
        include_response: bool = True
    ) -> str:
        """Format prompt in ChatML format for Qwen2.5."""
        tokens = get_qwen_tokens()
        user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
        
        prompt = f"{tokens['im_start']}system\n{self.system_prompt}{tokens['im_end']}\n"
        prompt += f"{tokens['im_start']}user\n{user_content}{tokens['im_end']}\n"
        prompt += f"{tokens['im_start']}assistant\n"
        
        if include_response and sql:
            prompt += f"{sql}{tokens['im_end']}"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        """Get stop tokens for generation."""
        return [get_qwen_tokens()["im_end"]]


@dataclass
class Phi3Template:
    """Prompt template for Phi-3-mini-instruct."""
    
    name: str = "phi3_mini"
    version: str = PROMPT_VERSION
    system_prompt: str = SYSTEM_PROMPT
    
    def format_prompt(
        self,
        schema: str,
        question: str,
        sql: Optional[str] = None,
        include_response: bool = True
    ) -> str:
        """Format prompt for Phi-3."""
        tokens = get_phi_tokens()
        user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
        
        prompt = f"{tokens['system']}\n{self.system_prompt} {tokens['end']}\n"
        prompt += f"{tokens['user']}\n{user_content} {tokens['end']}\n"
        prompt += f"{tokens['assistant']}\n"
        
        if include_response and sql:
            prompt += f"{sql} {tokens['end']}"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        """Get stop tokens for generation."""
        return [get_phi_tokens()["end"]]


def get_template(model_name: str):
    """Get the appropriate template for a model."""
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return Qwen25Template()
    elif "phi" in model_lower:
        return Phi3Template()
    else:
        # Default to Qwen template
        return Qwen25Template()


if __name__ == "__main__":
    # Test templates
    template = Qwen25Template()
    sample = template.format_prompt(
        schema="Table: users\nColumns: id, name, email",
        question="How many users are there?",
        sql="SELECT COUNT(*) FROM users"
    )
    print("=== Qwen2.5 Template ===")
    print(sample)
    print()
    
    template = Phi3Template()
    sample = template.format_prompt(
        schema="Table: users\nColumns: id, name, email",
        question="How many users are there?",
        sql="SELECT COUNT(*) FROM users"
    )
    print("=== Phi-3 Template ===")
    print(sample)
