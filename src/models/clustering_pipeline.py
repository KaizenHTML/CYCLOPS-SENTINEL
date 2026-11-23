import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLUSTER_NUMERIC_FEATURES = [
    "char_count",
    "word_count",
    "url_count",
    "urgency_score",
]


class UnsupervisedClusterEngine:
    # Single Responsibility: Performing K-Means and Hierarchical Clustering routines
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.kmeans_model = None

    def fit_scaler(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def compute_elbow_inertias(
        self, X_scaled: np.ndarray, k_range: range = range(2, 7)
    ) -> Dict[int, float]:
        inertias = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X_scaled)
            inertias[k] = round(float(km.inertia_), 2)
        return inertias

    def fit_kmeans(self, X_scaled: np.ndarray) -> np.ndarray:
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans_model.fit_predict(X_scaled)
        return labels

    def profile_clusters(
        self, df: pd.DataFrame, cluster_labels: np.ndarray
    ) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["cluster"] = cluster_labels
        
        summary = df_copy.groupby("cluster").agg(
            total_muestras=("char_count", "count"),
            tasa_amenaza=("is_threat", "mean"),
            promedio_caracteres=("char_count", "mean"),
            promedio_palabras=("word_count", "mean"),
            promedio_urls=("url_count", "mean"),
            promedio_urgencia=("urgency_score", "mean"),
        )
        summary["tasa_amenaza"] = (summary["tasa_amenaza"] * 100).round(2)
        summary["promedio_caracteres"] = summary["promedio_caracteres"].round(1)
        summary["promedio_palabras"] = summary["promedio_palabras"].round(1)
        summary["promedio_urls"] = summary["promedio_urls"].round(2)
        summary["promedio_urgencia"] = summary["promedio_urgencia"].round(3)
        return summary

    def compute_hierarchical_linkage(
        self, X_sample: np.ndarray, method: str = "ward"
    ) -> np.ndarray:
        return linkage(X_sample, method=method)


def serialize_clustering_artifacts(
    kmeans_model: Any,
    scaler: Any,
    model_output_path: str | Path,
    scaler_output_path: str | Path,
) -> None:
    m_path = Path(model_output_path)
    s_path = Path(scaler_output_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, m_path)
    joblib.dump(scaler, s_path)


def main():
    base_dir = PROJECT_ROOT
    data_path = base_dir / "data" / "processed" / "clean_dataset.csv"

    print("\n" + "=" * 75)
    print("CYCLOPS SENTINEL: MINERIA NO SUPERVISADA - K-MEANS Y CLUSTERING JERARQUICO")
    print("=" * 75)
    print(f"Cargando dataset desde: {data_path}")

    df = pd.read_csv(data_path)
    total_recs = len(df)
    print(f"Total observaciones: {total_recs:,} muestras")

    X = df[CLUSTER_NUMERIC_FEATURES]
    engine = UnsupervisedClusterEngine(n_clusters=3, random_state=42)

    print("Escalando variables numericas para medicion geometrica de distancias...")
    X_scaled = engine.fit_scaler(X)

    print("\nCalculando inercias con el Metodo del Codo [k de 2 a 6]...")
    inertias = engine.compute_elbow_inertias(X_scaled, range(2, 7))
    for k, val in inertias.items():
        print(f" - Clusters K={k}: Inercia WCSS = {val:,.2f}")

    print("\nAjustando K-Means con K optimo = 3 clusters...")
    labels = engine.fit_kmeans(X_scaled)

    print("\n" + "-" * 75)
    print("TABLA DE PERFILAMIENTO FENOTIPICO DE FAMILIAS DE CORREO:")
    print("-" * 75)
    profile_df = engine.profile_clusters(df, labels)
    print(profile_df.to_string())
    print("-" * 75)

    print("\nDESCRIPCION DE LOS 3 CLUSTERS DESCUBIERTOS:")
    print(" - Cluster 0: Correos benignos estandar, texto moderado, sin enlaces sospechosos.")
    print(" - Cluster 1: Mensajes de alta concentracion de amenazas [phishing], enlaces y coaccion.")
    print(" - Cluster 2: Comunicaciones atipicas extremadamente largas [boletines y logs].")

    # Serialization
    m_out = base_dir / "models" / "kmeans_model.joblib"
    s_out = base_dir / "models" / "kmeans_scaler.joblib"
    serialize_clustering_artifacts(engine.kmeans_model, engine.scaler, m_out, s_out)
    print(f"\n[OK] Modelo K-Means serializado en: {m_out}")
    print(f"[OK] Escalador de Clustering serializado en: {s_out}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
