import os
import json
import numpy as np
import pandas as pd
import joblib
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
from sklearn.neighbors import NearestNeighbors  # fallback if faiss not available


OUTPUTS_DIR = "outputs_metabric"
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "model_results")
INDEX_PATH = os.path.join(OUTPUTS_DIR, "patient_faiss.index")
PATIENT_META_PATH = os.path.join(OUTPUTS_DIR, "patient_similarity_meta.json")


class PatientSimilarityEngine:
    """Builds and queries a vector similarity index over all historical patients"""

    def __init__(self):
        self.index = None         
        self.patient_meta = []     
        self.X_embed = None        
        self._using_faiss = FAISS_AVAILABLE
        self.distance_scale = None 

    def build(self, force_rebuild = False) :
        """
        Build the similarity index from processed outputs
        """
        if not force_rebuild and self._load_cached():
            return True

        X_pca_path = os.path.join(OUTPUTS_DIR, "X_pca_20.csv")
        y_path = os.path.join(OUTPUTS_DIR, "y_labels.csv")

        if not os.path.exists(X_pca_path) or not os.path.exists(y_path):
            print("PatientSimilarityEngine: processed data not found. Run processing.py first.")
            return False

        X_embed = pd.read_csv(X_pca_path).values.astype("float32")
        y = pd.read_csv(y_path).iloc[:, 0].values

        raw_df = None
        raw_path = "data/METABRIC_RNA_Mutation.csv"
        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path, low_memory=False)
                raw_df.columns = (
                    raw_df.columns.str.strip().str.lower()
                    .str.replace(" ", "_").str.replace("+", "plus").str.replace("-", "_")
                )
            except Exception:
                raw_df = None

        self.X_embed = X_embed
        self.patient_meta = self._build_meta(y, raw_df)

        if self._using_faiss:
            d = X_embed.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(X_embed)
            faiss.write_index(self.index, INDEX_PATH)
        else:
            self.index = NearestNeighbors(metric="euclidean", algorithm="auto")
            self.index.fit(X_embed)

        with open(PATIENT_META_PATH, "w") as f:
            json.dump(self.patient_meta, f)

        self.distance_scale = self._estimate_distance_scale()
        print(f"PatientSimilarityEngine: indexed {len(y)} patients.")
        return True

    def find_similar(self, patient_index, k = 5) :
        """
        Find k most similar historical patients to the patient at patient_index
        """
        if self.index is None or self.X_embed is None:
            return []

        query = self.X_embed[[patient_index]].astype("float32")
        return self._search(query, k=k, exclude_index=patient_index)

    def find_similar_by_vector(self, query_vector, k = 10) :
        """
        Find similar historical patients for a custom query vector
        """
        if self.index is None or self.X_embed is None:
            return []
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return self._search(query, k=k, exclude_index=None)

    def get_similar_outcomes_summary(self, patient_index, k = 10) :
        """
        Summary of outcomes among the k most similar patients
        """
        similar = self.find_similar(patient_index, k=k)
        return self._summarize_results(similar)

    def get_similar_outcomes_summary_for_vector(self, query_vector, k = 10) :
        """
        Outcome summary for a custom patient feature vector
        """
        similar = self.find_similar_by_vector(query_vector, k=k)
        return self._summarize_results(similar)

    def _summarize_results(self, similar) :
        if not similar:
            return {"total": 0, "responders": 0, "non_responders": 0, "response_rate": None}

        outcomes = [s["outcome"] for s in similar if s["outcome"] in (0, 1)]
        responders = sum(1 for o in outcomes if o == 1)
        non_responders = len(outcomes) - responders

        return {
            "total": len(outcomes),
            "responders": responders,
            "non_responders": non_responders,
            "response_rate": round(responders / len(outcomes) * 100, 1) if outcomes else None,
            "patients": similar,
        }

    def _search(self, query, k, exclude_index) :
        k_query = min(k + (1 if exclude_index is not None else 0), len(self.X_embed))

        if self._using_faiss:
            distances, indices = self.index.search(query.astype("float32"), k_query)
            distances = distances[0]
            indices = indices[0]
        else:
            distances, indices = self.index.kneighbors(query.astype("float32"), n_neighbors=k_query)
            distances = distances[0]
            indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if exclude_index is not None and idx == exclude_index:
                continue
            if len(results) >= k:
                break

            meta = self.patient_meta[idx] if idx < len(self.patient_meta) else {}
            euclidean_dist = self._to_euclidean_distance(float(dist))
            similarity_pct = self._distance_to_similarity(euclidean_dist)

            results.append({
                "index": int(idx),
                "distance": round(euclidean_dist, 4),
                "similarity_pct": round(similarity_pct, 1),
                "outcome": int(meta.get("outcome", -1)),
                "outcome_label": "Alive at follow-up" if meta.get("outcome") == 1 else "Deceased",
                "age": meta.get("age", "N/A"),
                "er_status": meta.get("er_status", "N/A"),
                "her2_status": meta.get("her2_status", "N/A"),
                "tumor_stage": meta.get("tumor_stage", "N/A"),
                "treatments": meta.get("treatments", "N/A"),
            })

        return results

    def _build_meta(self, y, raw_df) :
        meta = []
        for i, outcome in enumerate(y):
            entry = {"outcome": int(outcome)}
            if raw_df is not None and i < len(raw_df):
                row = raw_df.iloc[i]
                entry["age"] = _safe(row.get("age_at_diagnosis"))
                entry["er_status"] = _safe(row.get("er_status"))
                entry["her2_status"] = _safe(row.get("her2_status"))
                entry["tumor_stage"] = _safe(row.get("tumor_stage"))
                chemo = _safe(row.get("chemotherapy"))
                hormone = _safe(row.get("hormone_therapy"))
                radio = _safe(row.get("radio_therapy"))
                txs = []
                if str(chemo) in ("1", "1.0"):
                    txs.append("Chemo")
                if str(hormone) in ("1", "1.0"):
                    txs.append("Hormone")
                if str(radio) in ("1", "1.0"):
                    txs.append("Radiation")
                entry["treatments"] = ", ".join(txs) if txs else "None recorded"
            meta.append(entry)
        return meta

    def _load_cached(self) :
        """
        Try to load a previously built index from disk
        """
        if not os.path.exists(PATIENT_META_PATH):
            return False

        pca_path = os.path.join(OUTPUTS_DIR, "X_pca_20.csv")
        if not os.path.exists(pca_path):
            return False

        try:
            with open(PATIENT_META_PATH) as f:
                self.patient_meta = json.load(f)

            self.X_embed = pd.read_csv(pca_path).values.astype("float32")

            if self._using_faiss and os.path.exists(INDEX_PATH):
                self.index = faiss.read_index(INDEX_PATH)
            else:
                self.index = NearestNeighbors(metric="euclidean", algorithm="auto")
                self.index.fit(self.X_embed)
                self._using_faiss = False

            self.distance_scale = self._estimate_distance_scale()
            return True
        except Exception:
            return False

    def _to_euclidean_distance(self, raw_distance) :
        """
        Convert a raw distance to Euclidean distance
        If using FAISS IndexFlatL2 returns squared L2 distance
        Else sklearn returns Euclidean distance
        """
        if self._using_faiss:
            return float(np.sqrt(max(raw_distance, 0.0)))
        return raw_distance

    def _estimate_distance_scale(self, sample_size = 300) :
        """
        Estimates a typical distance value between similar patients in the dataset
        """
        if self.index is None or self.X_embed is None or len(self.X_embed) < 2:
            return 1.0

        n = min(sample_size, len(self.X_embed))
        sample = self.X_embed[:n].astype("float32")

        if self._using_faiss:
            distances, _ = self.index.search(sample, 2)
            nn_distances = distances[:, 1]
        else:
            distances, _ = self.index.kneighbors(sample, n_neighbors=2)
            nn_distances = distances[:, 1]

        euclidean = np.array([self._to_euclidean_distance(float(d)) for d in nn_distances], dtype=float)
        scale = float(np.percentile(euclidean, 95))
        return scale if scale > 0 else 1.0

    def _distance_to_similarity(self, distance) :
        """
        Convert Euclidean distance to a bounded similarity percentage
        """
        scale = self.distance_scale or 1.0
        similarity = 100.0 * np.exp(-((distance / scale) ** 2))
        return min(max(float(similarity), 0.0), 99.9)


def _safe(val) :
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return str(val).strip()

_engine_instance = None

def get_engine() :
    """ 
    Return a shared, built engine instance 
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PatientSimilarityEngine()
        _engine_instance.build()
    return _engine_instance
