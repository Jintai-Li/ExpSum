import json
import re
import inspect
import numpy as np

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, issparse


class ContextAwareKnowledgeBase:
    def __init__(self, filepath: str = None):
        self.vectorizer = TfidfVectorizer(min_df=1, token_pattern=r'(?u)\b\w+\b')
        self.knowledge = []
        self.term_index = defaultdict(set)
        self.vectorizer = TfidfVectorizer()
        self.filepath = Path(filepath) if filepath else None
        self._dirty = True

        if filepath and self.filepath.exists():
            self.load(filepath)

    def _safe_fit(self, contexts: List[str]):
        if isinstance(self.vectorizer.dtype, str):
            if 'float64' in self.vectorizer.dtype:
                self.vectorizer.dtype = np.float64
        self.vectorizer.fit(contexts)

    def add_entry(self, context: str, terms: Dict[str, str]):
        if not re.search(r'\w', context):
            raise ValueError("Context must contain valid tokens")

        if any(entry["context"] == context for entry in self.knowledge):
            return

        entry = {
            "context": context,
            "terms": terms,
            "vector": None
        }
        self.knowledge.append(entry)

        for term in terms:
            self.term_index[term].add(len(self.knowledge) - 1)

        self._dirty = True

    def _rebuild_vectors(self):
        contexts = [entry["context"] for entry in self.knowledge]

        if not contexts:
            self._dirty = False
            return

        if not hasattr(self.vectorizer, "vocabulary_"):
            self._safe_fit(contexts)

        vectors = self.vectorizer.transform(contexts)

        for i, vec in enumerate(vectors):
            self.knowledge[i]["vector"] = csr_matrix(vec)

        self._dirty = False

    def search(self, query: str, top_n: int = 9, similarity_threshold: float = 0):
        if self._dirty:
            self._rebuild_vectors()

        if not hasattr(self.vectorizer, "vocabulary_") or not self.vectorizer.vocabulary_:
            self._rebuild_vectors()

        try:
            query_vec = self.vectorizer.transform([query])
        except ValueError:
            vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0
            query_vec = csr_matrix((1, vocab_size))

        candidate_terms = self._fuzzy_term_match(query)
        candidate_indices = set()
        for term in candidate_terms:
            candidate_indices.update(self.term_index.get(term, set()))

        similarities = []
        for idx in candidate_indices:
            entry = self.knowledge[idx]
            entry_vec = entry["vector"]

            if isinstance(entry_vec, np.ndarray):
                entry_vec = csr_matrix(entry_vec.reshape(1, -1))
            elif not issparse(entry_vec):
                entry_vec = csr_matrix(entry_vec)

            dot_product = entry_vec.dot(query_vec.T)[0, 0]
            norm_entry = np.sqrt(entry_vec.power(2).sum())
            norm_query = np.sqrt(query_vec.power(2).sum())
            sim = dot_product / (norm_entry * norm_query + 1e-8)

            if sim >= similarity_threshold:
                similarities.append((sim, idx))

        similarities.sort(key=lambda x: -x[0])
        top_results = similarities[:min(top_n, len(similarities))]

        term_pool = defaultdict(list)
        for sim, idx in top_results:
            entry = self.knowledge[idx]
            valid_terms = {
                cn: en for cn, en in entry["terms"].items()
                if cn in candidate_terms
            }
            for cn, en in valid_terms.items():
                term_pool[cn].append((en, sim))

        final_results = []
        for cn, candidates in term_pool.items():
            best_trans = max(candidates, key=lambda x: x[1])[0]
            final_results.append({
                "source_term": cn,
                "target_term": best_trans
            })

        return final_results

    def batch_import(self, import_path: str):
        with open(import_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must be a list of dictionaries")

        stats = {'added': 0, 'skipped': 0, 'errors': 0}

        for idx, item in enumerate(data, 1):
            try:
                if 'context' not in item or 'terms' not in item:
                    raise ValueError("Missing required fields")

                if any(entry['context'] == item['context'] for entry in self.knowledge):
                    stats['skipped'] += 1
                    continue

                self.add_entry(item['context'], item['terms'])
                stats['added'] += 1
                self._dirty = True
            except Exception:
                stats['errors'] += 1

        return stats

    def delete_entries(self, context: str):
        deleted = 0
        for i in range(len(self.knowledge) - 1, -1, -1):
            if self.knowledge[i]['context'] == context:
                for term in self.knowledge[i]['terms']:
                    self.term_index[term].discard(i)
                    for t in self.term_index:
                        self.term_index[t] = {
                            idx if idx < i else idx - 1
                            for idx in self.term_index[t]
                        }
                del self.knowledge[i]
                deleted += 1
                self._dirty = True
        return deleted

    def _fuzzy_term_match(self, text: str):
        matched_terms = set()
        tokenized = set(re.findall(r'[\w\u4e00-\u9fff]+', text))

        for term in self.term_index:
            if term in text:
                matched_terms.add(term)
                continue

            term_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', term))
            if len(term_tokens & tokenized) / len(term_tokens) >= 0.5:
                matched_terms.add(term)

        return matched_terms

    def save(self, filepath: str = None):
        save_path = Path(filepath) if filepath else self.filepath
        if not save_path:
            raise ValueError("Storage path must be specified")

        if self._dirty:
            self._rebuild_vectors()

        serializable_params = {}
        for k, v in self.vectorizer.get_params().items():
            if inspect.isclass(v):
                serializable_params[k] = f"CLASS:{v.__module__}.{v.__name__}"
            elif callable(v):
                serializable_params[k] = f"FUNCTION:{v.__name__}"
            else:
                serializable_params[k] = v

        save_data = {
            "knowledge": [
                {
                    "context": entry["context"],
                    "terms": entry["terms"],
                    "vector_data": entry["vector"].toarray().flatten().tolist(),
                    "vector_shape": list(entry["vector"].shape)
                }
                for entry in self.knowledge
            ],
            "term_index": {k: list(v) for k, v in self.term_index.items()},
            "vectorizer_params": serializable_params
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=self._json_default)

    def _json_default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if inspect.isclass(obj):
            return f"CLASS:{obj.__module__}.{obj.__name__}"
        if callable(obj):
            return f"FUNCTION:{obj.__name__}"
        raise TypeError(f"Object not serializable: {obj}")

    def load(self, filepath: str):
        path = Path(filepath)
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        params = save_data.get("vectorizer_params", {})
        if 'ngram_range' in params and isinstance(params['ngram_range'], list):
            params['ngram_range'] = tuple(params['ngram_range'])

        self.vectorizer = TfidfVectorizer(**params)

        self.knowledge = []
        for entry_data in save_data["knowledge"]:
            if entry_data["vector_data"]:
                dense_array = np.array(entry_data["vector_data"])
                vector = csr_matrix(dense_array.reshape(entry_data["vector_shape"]))
            else:
                vector = csr_matrix((0, 0))

            self.knowledge.append({
                "context": entry_data["context"],
                "terms": entry_data["terms"],
                "vector": vector
            })

        self.term_index = defaultdict(set)
        for term, indices in save_data["term_index"].items():
            self.term_index[term] = set(indices)

        self._dirty = True
