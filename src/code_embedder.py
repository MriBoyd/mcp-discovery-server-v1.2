# src/code_embedder.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
from src.config import Config
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.model_path = model_path
        self.cache_dir = cache_dir
        
        logger.info(f"Loading CodeEmbedder from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir=cache_dir,
            local_files_only=True  # Force offline
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            local_files_only=True
        )
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        logger.info(f"Model loaded on {Config.DEVICE}")
    
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
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().to(torch.float32).numpy().tolist()
    
    def encode_tool(self, tool_text: str) -> List[float]:
        """Encode a tool for indexing (as passage)"""
        return self.encode(tool_text, task="nl2code", is_query=False)[0]
    
    def encode_query(self, query: str) -> List[float]:
        """Encode a user query for searching"""
        return self.encode(query, task="nl2code", is_query=True)[0]
    
    def batch_encode_tools(self, tool_texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch encode multiple tools"""
        all_embeddings = []
        for i in range(0, len(tool_texts), batch_size):
            batch = tool_texts[i:i+batch_size]
            embeddings = self.encode(batch, task="nl2code", is_query=False)
            all_embeddings.extend(embeddings)
        return all_embeddings