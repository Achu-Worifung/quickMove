from util.util import resource_path
import os
import json
import re   
import time 
import numpy as np
import torch
from scipy.special import expit
import faiss
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from optimum.onnxruntime import ORTModelForSequenceClassification

class BibleSearch:
    """Search functionality for Bible verses"""
    
    def __init__(self, device_type="cpu"):
        

        self.device_type = device_type
        self.device = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
        
        # Configuration
        self.DATA_FILE = resource_path("bibles/merged_bible.json")
        self.ARTIFACT_DIR = resource_path("artifacts")
        
        # Ensure artifact directory exists
        os.makedirs(self.ARTIFACT_DIR, exist_ok=True)
        
        # File paths
        self.EMBEDDINGS_FILE = f"{self.ARTIFACT_DIR}/embeddings.npy"
        self.FAISS_FILE = f"{self.ARTIFACT_DIR}/faiss.index"
        self.METADATA_FILE = f"{self.ARTIFACT_DIR}/metadata.json"
        self.TEXTS_FILE = f"{self.ARTIFACT_DIR}/texts.json"
        
        # Model names
        self.MODEL_NAME = 'all-mpnet-base-v2'
        self.MODEL_PATH = resource_path(f"./search/{self.MODEL_NAME}")
        self.RE_RANKER_MODEL = 'cross-encoder_ms-marco-MiniLM-L6-v2'
        self.ONNX_MODEL_DIR = resource_path(f"./search/{self.RE_RANKER_MODEL.replace('/', '_')}_onnx")
        
        # Search parameters
        self.BM25_TOP_K = 5000
        self.SEMANTIC_TOP_K = 100
        self.FINAL_TOP_K = 10
        self.BATCH_SIZE = 32
        
        # Initialize components
        self.texts = None
        self.metadata = None
        self.bm25 = None
        self.embeddings = None
        self.faiss_index = None
        self.model = None
        self.tokenizer = None
        self.onnx_reranker = None
        self.reranker_tokenizer = None
        
        # Load data and models
        self.load_data_and_models()
    
    def tokenize(self, text):
        """Tokenize text for BM25"""
        return re.findall(r"\b\w+\b", text.lower())
    
    def load_data_and_models(self):
        """Load all data and models for search"""
        print("📖 Loading Bible search data...")
        
        # Load Bible data
        with open(self.DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.texts = [v["text"] for v in data]
        self.metadata = [
            {"book": v["book"], "chapter": v["chapter"], "verse": v["verse"], "version": v["version"]}
            for v in data
        ]
        
        # Build BM25 index
        print("⚙️ Building BM25 index...")
        tokenized_texts = [self.tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Load embedding model
        print(f"Using device: {self.device}")
        model_path = self.MODEL_PATH
        
        try:
            if os.path.exists(model_path):
                print(f" Loading cached embedding model from {model_path}...")
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            else:
                print(f"⬇️ Downloading embedding model {self.MODEL_NAME}...")
                self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)
        except Exception as e:
            print(f" Failed to load embedding model: {e}")
            raise
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.MODEL_NAME}")
        self.model.to(self.device)
        self.model.eval()
        
        # Load ONNX reranker
        self.load_onnx_reranker()
        
        # Load or build embeddings and FAISS index
        self.load_or_build_embeddings()
        
        print("✅ Bible search initialized successfully!")
    
    def load_onnx_reranker(self):
        """Load or convert cross-encoder to ONNX format"""
        print("📥 Loading or converting ONNX reranker model...")
        if os.path.exists(self.ONNX_MODEL_DIR):
            print(f"✅ Loading ONNX reranker from {self.ONNX_MODEL_DIR}...")
            self.onnx_reranker = ORTModelForSequenceClassification.from_pretrained(
                self.ONNX_MODEL_DIR,
                provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.ONNX_MODEL_DIR)
        else:
            print(f"⚙️ Converting {self.RE_RANKER_MODEL} to ONNX (one-time setup)...")
            self.onnx_reranker = ORTModelForSequenceClassification.from_pretrained(
                resource_path(self.RE_RANKER_MODEL),
                export=True,
                provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(resource_path(self.RE_RANKER_MODEL))
            
            os.makedirs(self.ONNX_MODEL_DIR, exist_ok=True)
            self.onnx_reranker.save_pretrained(self.ONNX_MODEL_DIR)
            self.reranker_tokenizer.save_pretrained(self.ONNX_MODEL_DIR)
            print(f"✅ Saved ONNX model to {self.ONNX_MODEL_DIR}")
    
    def load_or_build_embeddings(self):
        print("📥 Loading or building embeddings and FAISS index...")
        """Load embeddings and FAISS index from disk or build them"""
        if os.path.exists(self.EMBEDDINGS_FILE) and os.path.exists(self.FAISS_FILE):
            print("✅ Loading embeddings and FAISS index from disk...")
            self.embeddings = np.load(self.EMBEDDINGS_FILE)
            self.faiss_index = faiss.read_index(self.FAISS_FILE)
            
            with open(self.METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            with open(self.TEXTS_FILE, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
        else:
            print("⚙️ Building embeddings and FAISS index (one-time)...")
            self.embeddings = self.get_embeddings(self.texts, batch_size=self.BATCH_SIZE).astype("float32")
            
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings)
            
            # Save to disk
            np.save(self.EMBEDDINGS_FILE, self.embeddings)
            faiss.write_index(self.faiss_index, self.FAISS_FILE)
            
            with open(self.METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            
            with open(self.TEXTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.texts, f)
            
            print("✅ Saved embeddings and FAISS index")
    
    def get_embeddings(self, texts, batch_size=32):
        """Get embeddings for a list of texts"""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(emb)
        return np.vstack(all_embs)
    
    def bm25_recall(self, query, top_k=5000):
        """BM25 keyword recall stage"""
        start = time.time()
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        end = time.time()
        print(f"  BM25 recall: {end - start:.4f}s")
        return top_idx, scores[top_idx]
    
    def semantic_rerank(self, query, candidate_indices, top_k=100):
        """Semantic reranking stage using embeddings"""
        start = time.time()
        q_emb = self.get_embeddings([query])
        faiss.normalize_L2(q_emb)
        
        candidate_embs = self.embeddings[candidate_indices]
        faiss.normalize_L2(candidate_embs)
        
        scores = candidate_embs @ q_emb.T
        scores = scores.squeeze()
        
        top = np.argsort(scores)[::-1][:top_k]
        top_indices = candidate_indices[top]
        
        end = time.time()
        print(f"  Semantic rerank: {end - start:.4f}s")
        return top_indices

    
    def rerank_with_crossencoder(self, query, candidate_indices, top_k=10):
        """Final reranking with cross-encoder"""
        start = time.time()
        
        if len(candidate_indices) == 0:
            return []
        
        candidate_texts = [self.texts[idx] for idx in candidate_indices]
        
        # Tokenize inputs for ONNX model
        inputs = self.reranker_tokenizer(
            [query] * len(candidate_texts),
            candidate_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device if using GPU
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get scores from ONNX model
        with torch.no_grad():
            outputs = self.onnx_reranker(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
            scores = expit(scores)  # Apply sigmoid to convert logits to probabilities
        
        # Get top-k results
        top = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top:
            idx = candidate_indices[i]
            meta = self.metadata[idx]
            results.append({
                "score": float(scores[i]),
                "ref": f"{meta['book']} {meta['chapter']}:{meta['verse']} ({meta['version']})",
                "text": self.texts[idx],
                "book": meta["book"],
                "chapter": meta["chapter"],
                "verse": meta["verse"],
                "version": meta["version"]
            })
        
        end = time.time()
        print(f"  Cross-encoder ONNX: {end - start:.4f}s")
        
        return results
    
    def search(self, query, bm25_top_k=None, semantic_top_k=None, final_top_k=None):
        """Main search pipeline"""
        print(f"\n🔍 Searching for: '{query}'")
        start_total = time.time()
        
        # Use provided parameters or defaults
        bm25_top_k = bm25_top_k or self.BM25_TOP_K
        semantic_top_k = semantic_top_k or self.SEMANTIC_TOP_K
        final_top_k = final_top_k or self.FINAL_TOP_K
        
        # Stage 1: BM25 keyword recall
        candidate_idx, _ = self.bm25_recall(query, top_k=bm25_top_k)
        
        # Stage 2: Semantic rerank
        semantic_top_indices = self.semantic_rerank(query, candidate_idx, top_k=semantic_top_k)
        
        
        # Stage 3: Cross-encoder final rerank
        final_results = self.rerank_with_crossencoder(query, semantic_top_indices, top_k=final_top_k)
        
        total_time = time.time() - start_total
        
        
        return final_results
