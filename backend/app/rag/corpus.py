"""Local exact-cosine corpus shared by runtime and evaluation."""
import json
import numpy as np
from rank_bm25 import BM25Okapi
from backend.app.rag.retrieval_common import _payload_matches_scope, _tokenize


class OfflineCorpus:
    def __init__(self, path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows or len({str(r['id']) for r in self.rows}) != len(self.rows):
            raise ValueError("Corpus is empty or contains duplicate IDs")
        texts = [" ".join(str(row["payload"].get(key) or "") for key in
                 ("program_name", "program_code", "section", "text")) for row in self.rows]
        self.bm25 = BM25Okapi([_tokenize(text) for text in texts])
        self.vectors = np.asarray([row["vector"] for row in self.rows], dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if not np.isfinite(self.vectors).all() or np.any(norms == 0):
            raise ValueError("Corpus contains invalid dense vectors")
        self.vectors /= norms

    def dense_search(self, vector, scope, k):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all() or norm == 0:
            raise ValueError("Query vector is invalid or incompatible with cached corpus")
        eligible = [i for i, row in enumerate(self.rows) if _payload_matches_scope(row["payload"], scope)]
        if not eligible:
            return []
        scores = self.vectors[eligible] @ (vector / norm)
        top = sorted(range(len(eligible)), key=lambda j: (-float(scores[j]), str(self.rows[eligible[j]]["id"])))[:k]
        return [{**self.rows[eligible[j]], "score": float(scores[j])} for j in top]

