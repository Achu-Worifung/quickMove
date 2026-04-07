import os
import json
import re
import time
import numpy as np
import faiss
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification


class BibleSearch:
    """High-precision Bible semantic search using hybrid sparse + dense + rerank pipeline"""

    def __init__(self):
        # =========================
        # CONFIG
        # =========================
        self.DATA_FILE = "bibles/merged_cleaned.json"
        self.ARTIFACT_DIR = "artifacts"

        self.EMBEDDINGS_FILE = f"{self.ARTIFACT_DIR}/embeddings.npy"
        self.FAISS_FILE = f"{self.ARTIFACT_DIR}/faiss_hnsw.index"
        self.ONNX_EMBED_DIR = f"{self.ARTIFACT_DIR}/onnx_embedder"
        self.ONNX_RERANK_DIR = f"{self.ARTIFACT_DIR}/onnx_reranker"
        self.EMBED_TOKENIZER = f"{self.ARTIFACT_DIR}/embed_tokenizer"
        self.INVERTED_INDEX_FILE = f"{self.ARTIFACT_DIR}/inverted_index.json"

        # Local model paths (run download_models.py once to populate these)
        self.RERANKER_MODEL = "search/ms-marco-MiniLM-L4-v2"
        self.SEMANTIC_MODEL = "search/multi-qa-MiniLM-L6-cos-v1"

        self.SEMANTIC_TOP_K = 20
        self.KEYWORD_MERGE_K = 30
        self.FINAL_TOP_K = 10
        self.BATCH_SIZE = 32
        self.KEYWORD_ONLY_MAX = 3

        self.DEVICE = "cpu"

        os.makedirs(self.ARTIFACT_DIR, exist_ok=True)

        # Internal state
        self.texts = []
        self.metadata = []
        self.inverted_index = {}
        self.embed_tokenizer = None
        self.embed_model = None
        self.index = None
        self.reranker_tokenizer = None
        self.onnx_model = None

        # Boot up
        self._load_data()
        self._load_inverted_index()
        self._load_embedder()
        self._load_faiss()
        self._load_reranker()
        self._warmup()

    # =========================
    # UTILS
    # =========================

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _safe_indices(self, indices):
        """Filter out any indices that are out of bounds for self.texts"""
        max_idx = len(self.texts)
        valid = [int(i) for i in indices if 0 <= int(i) < max_idx]
        filtered = len(indices) - len(valid)
        if filtered > 0:
            print(f"  ⚠️  Filtered {filtered} out-of-range indices (texts={max_idx})")
        return valid

    # =========================
    # LOAD DATA
    # =========================

    def _load_data(self):
        print("Loading Bible data...")
        with open(self.DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = [v for v in data if v["text"].strip()]
        self.texts = [v["text"] for v in data]
        self.metadata = [
            {"book": v["book"], "chapter": v["chapter"], "verse": v["verse"], "version": v["version"]}
            for v in data
        ]
        print(f"Loaded {len(self.texts)} verses after filtering empty entries")

    # =========================
    # INVERTED INDEX
    # =========================

    def _load_inverted_index(self):
        if os.path.exists(self.INVERTED_INDEX_FILE):
            print("Loading cached inverted index...")
            with open(self.INVERTED_INDEX_FILE, "r", encoding="utf-8") as f:
                self.inverted_index = {k: set(v) for k, v in json.load(f).items()}
        else:
            print("Building inverted index...")
            index = defaultdict(set)
            for i, text in enumerate(self.texts):
                for token in set(self._tokenize(text)):
                    index[token].add(i)
            self.inverted_index = dict(index)
            with open(self.INVERTED_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in self.inverted_index.items()}, f, indent=2)

        print(f"Inverted index ready: {len(self.inverted_index)} unique tokens")

    def _keyword_recall(self, query):
        tokens = self._tokenize(query)
        if not tokens:
            return np.array([], dtype=np.int64)
        candidates = self.inverted_index.get(tokens[0], set()).copy()
        for t in tokens[1:]:
            candidates &= self.inverted_index.get(t, set())
            if not candidates:
                break
        return np.array(list(candidates), dtype=np.int64)

    # =========================
    # EMBEDDING MODEL
    # =========================

    def _load_embedder(self):
        print("Loading ONNX embedding model...")

        if os.path.exists(self.EMBED_TOKENIZER):
            print("  Loading cached embed tokenizer...")
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.EMBED_TOKENIZER)
        else:
            print("  Saving embed tokenizer from local model (one-time)...")
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.SEMANTIC_MODEL)
            self.embed_tokenizer.save_pretrained(self.EMBED_TOKENIZER)
            print(f"  Saved to {self.EMBED_TOKENIZER}")

        if os.path.exists(self.ONNX_EMBED_DIR):
            print("  Loading cached ONNX embedder...")
            self.embed_model = ORTModelForFeatureExtraction.from_pretrained(
                self.ONNX_EMBED_DIR,
                provider="CPUExecutionProvider"
            )
        else:
            print("  Exporting embedder to ONNX (one-time)...")
            self.embed_model = ORTModelForFeatureExtraction.from_pretrained(
                self.SEMANTIC_MODEL,
                export=True,
                provider="CPUExecutionProvider"
            )
            self.embed_model.save_pretrained(self.ONNX_EMBED_DIR)
            print(f"  Saved to {self.ONNX_EMBED_DIR}")

    def _get_embeddings(self, texts_batch):
        all_embs = []
        for i in range(0, len(texts_batch), self.BATCH_SIZE):
            batch = texts_batch[i:i + self.BATCH_SIZE]
            inputs = self.embed_tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.DEVICE)
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = emb / emb.norm(dim=1, keepdim=True)
                all_embs.append(emb)
        return torch.cat(all_embs, dim=0)

    # =========================
    # FAISS HNSW INDEX
    # =========================

    def _load_faiss(self):
        if os.path.exists(self.EMBEDDINGS_FILE) and os.path.exists(self.FAISS_FILE):
            print("Loading embeddings + FAISS index...")
            self.index = faiss.read_index(self.FAISS_FILE)
            faiss_size = self.index.ntotal
            texts_size = len(self.texts)
            if faiss_size != texts_size:
                print(f"  ⚠️  FAISS index size ({faiss_size}) != texts size ({texts_size})")
                print("  Rebuilding index to match current dataset...")
                self._build_faiss()
            else:
                print(f"  FAISS index verified: {faiss_size} vectors")
        else:
            self._build_faiss()

    def _build_faiss(self):
        print("Building embeddings + HNSW index (one-time)...")
        embeddings_cpu = self._get_embeddings(self.texts).cpu().numpy()
        dim = embeddings_cpu.shape[1]

        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 32
        self.index.add(embeddings_cpu)

        np.save(self.EMBEDDINGS_FILE, embeddings_cpu)
        faiss.write_index(self.index, self.FAISS_FILE)
        print(f"Index built and saved ({len(self.texts)} vectors).")

    # =========================
    # RERANKER
    # =========================

    def _load_reranker(self):
        print("Loading ONNX reranker...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.RERANKER_MODEL)

        if os.path.exists(self.ONNX_RERANK_DIR):
            print("  Loading cached ONNX reranker...")
            self.onnx_model = self._load_ort_model(self.ONNX_RERANK_DIR)
        else:
            print("  Exporting reranker to ONNX (one-time)...")
            self.onnx_model = self._load_ort_model(self.RERANKER_MODEL, export=True)
            self.onnx_model.save_pretrained(self.ONNX_RERANK_DIR)
            print(f"  Saved to {self.ONNX_RERANK_DIR}")

        self.onnx_model.use_io_binding = False  # safe on Windows

    def _load_ort_model(self, model_id, export=False):
        try:
            print("  Trying CUDAExecutionProvider...")
            model = ORTModelForSequenceClassification.from_pretrained(
                model_id, export=export, provider="CUDAExecutionProvider"
            )
            print("  Reranker loaded on CUDA.")
            return model
        except Exception as e:
            print(f"  CUDA failed ({e}), falling back to CPU...")
            model = ORTModelForSequenceClassification.from_pretrained(
                model_id, export=export, provider="CPUExecutionProvider"
            )
            print("  Reranker loaded on CPU.")
            return model

    # =========================
    # WARMUP
    # =========================

    def _warmup(self):
        print("Warming up models...")
        self._get_embeddings(["warmup"])
        _wr = self.reranker_tokenizer(
            ["warmup"], ["warmup"], return_tensors="pt", padding=True, truncation=True
        ).to(self.DEVICE)
        self.onnx_model(**_wr)
        print("Ready!\n")

    # =========================
    # PIPELINE
    # =========================

    def _semantic_search(self, query):
        q_emb = self._get_embeddings([query]).cpu().numpy()
        D, I = self.index.search(q_emb, self.SEMANTIC_TOP_K)
        return I[0]

    def _format_results(self, indices, scores=None):
        seen_refs = set()
        results = []
        for rank, idx in enumerate(indices):
            meta = self.metadata[idx]
            ref_key = f"{meta['book']} {meta['chapter']}:{meta['verse']}"
            display_ref = f"{ref_key} [{meta['version']}]"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            results.append({
                "score": float(scores[rank]) if scores is not None else 1.0,
                "ref": display_ref,
                "text": self.texts[idx],
                "book": meta["book"],
                "chapter": meta["chapter"],
                "verse": meta["verse"],
                "version": meta["version"]
            })
            if len(results) >= self.FINAL_TOP_K:
                break
        return results

    def _rerank(self, query, candidate_indices):
        # Guard: filter out any stale/out-of-range indices before touching self.texts
        candidate_indices = self._safe_indices(candidate_indices)
        if not candidate_indices:
            return []

        candidate_texts = [self.texts[i] for i in candidate_indices]
        inputs = self.reranker_tokenizer(
            [query] * len(candidate_texts), candidate_texts,
            padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        outputs = self.onnx_model(**inputs)
        scores = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
        top = np.argsort(scores)[::-1]
        ranked_indices = [candidate_indices[i] for i in top]
        ranked_scores = scores[top]
        return self._format_results(ranked_indices, ranked_scores)

    def search(self, query, final_top_k=10):
        """Main search entry point. Returns list of ranked verse results."""
        start = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            kw_future = executor.submit(self._keyword_recall, query)
            sem_future = executor.submit(self._semantic_search, query)
            keyword_candidates = kw_future.result()
            semantic_candidates = sem_future.result()

        t_parallel = time.time()

        # Guard: sanitize all candidates before any index access
        keyword_candidates = self._safe_indices(keyword_candidates)
        semantic_candidates = self._safe_indices(semantic_candidates)
        
        # this is for testing only remove latter on ok 
        kw_dict = self._format_results(keyword_candidates)
        print("===" * 50)
        for kw in kw_dict:
            print(kw)
        print("===" * 50)
        sm_dict = self._format_results(semantic_candidates)
        for sm in sm_dict:
            print(sm)

        
        if 0 < len(keyword_candidates) <= self.KEYWORD_ONLY_MAX:
            results = self._format_results(keyword_candidates)
            path = f"keyword-only ({len(keyword_candidates)} hits)"
            t_end = time.time()
        else:
            merged = list(dict.fromkeys(
                list(semantic_candidates) + list(keyword_candidates[:self.KEYWORD_MERGE_K])
            ))
            results = self._rerank(query, merged)
            t_end = time.time()
            path = (
                f"semantic+rerank (merged={len(merged)}, "
                f"keyword={len(keyword_candidates)}, "
                f"semantic={len(semantic_candidates)})"
            )

        total = t_end - start
        print(
            f"\n⚡ Recall: {(t_parallel - start) * 1000:.1f}ms | "
            f"Rerank: {(t_end - t_parallel) * 1000:.1f}ms | "
            f"Total: {total * 1000:.1f}ms | Path: {path}"
        )
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['ref']} (score={r['score']:.3f})")
            print(f"   {r['text']}")
        return results


# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    searcher = BibleSearch()
    while True:
        q = input("\nEnter query (or 'exit'): ")
        if q.lower() == "exit":
            break
        results = searcher.search(q)
        print("\nResults:")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['ref']} (score={r['score']:.3f})")
            print(f"   {r['text']}")
        print("-" * 60)