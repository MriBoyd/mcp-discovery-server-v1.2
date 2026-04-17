# src/code_embedder.py
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
from src.config import Config
import logging 

from functools import lru_cache

class CodeEmbedder:
    """
    Local wrapper for jina-code-embeddings-1.5b
    Supports task-specific instructions for optimal retrieval
    """
    
    INSTRUCTION_CONFIG = {
        "nl2code": {
            "query": "Find the most relevant code snippet given the following query:\n",
            "passage": "Candidate code snippet:\n"
        },
        "code2code": {
            "query": "Find an equivalent code snippet given the following code snippet:\n",
            "passage": "Candidate code snippet:\n"
        },
        "qa": {
            "query": "Find the most relevant answer given the following question:\n",
            "passage": "Candidate answer:\n"
        }
    }
    
    def __init__(self, model_path: str = "./embed-models", 
                 cache_dir: str = "."):
        
        # Optimization: Set threads for CPU inference if not already set by reranker
        if Config.DEVICE == "cpu":
            import os
            # Use a reasonable number of threads
            num_threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(num_threads)

        current_file_dir = Path(__file__).parent.resolve()
        project_root = current_file_dir.parent
        abs_model_path = (project_root / model_path).resolve()
        
        self.model_path = str(abs_model_path)
        
        if not abs_model_path.exists():
            # List directory content to help you debug in logs
            raise FileNotFoundError(f"Could not find model at {abs_model_path}")
        
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir=cache_dir,
            local_files_only=True  # Force offline
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            dtype=torch.bfloat16 if Config.DEVICE == "cuda" else torch.float32,
            local_files_only=True
        )
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        # CPU Optimization: Dynamic Quantization
        if Config.DEVICE == "cpu":
            try:
                # Quantize Linear layers to int8 for speedup on CPU
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to quantize embedder: {e}")
        
    
    def last_token_pool(self, last_hidden_states, attention_mask):
        """Last token pooling (Jina Code Embeddings uses this)"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def encode(self, texts: Union[str, List[str]], task: str = "nl2code", 
               is_query: bool = True) -> List[List[float]]:
        """
        Encode texts with task-specific instruction
        
        Args:
            texts: Single text or list of texts
            task: "nl2code", "code2code", or "qa"
            is_query: True for queries, False for passages
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Add instruction prefix
        instruction_key = "query" if is_query else "passage"
        prefix = self.INSTRUCTION_CONFIG[task][instruction_key]
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        
        # Tokenize
        batch_dict = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        
        # Encode
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().to(torch.float32).numpy().tolist()
    
    @lru_cache(maxsize=128)
    def encode_query(self, query: str) -> List[float]:
        """Encode a user query for searching (cached)"""
        return self.encode(query, task="nl2code", is_query=True)[0]
    
    def batch_encode_tools(self, tool_texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch encode multiple tools"""
        all_embeddings = []
        for i in range(0, len(tool_texts), batch_size):
            batch = tool_texts[i:i+batch_size]
            embeddings = self.encode(batch, task="nl2code", is_query=False)
            all_embeddings.extend(embeddings)
        return all_embeddings