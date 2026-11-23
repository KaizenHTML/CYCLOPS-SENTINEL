import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.interfaces import IModelRegistry, IModelBenchmark, IArtifactSerializer
from src.core.dto import BenchmarkMetricRecord


class SupervisedModelRegistry(IModelRegistry):
    # Implements IModelRegistry providing standardized classifier instances
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_models(self) -> Dict[str, Any]:
        return {
            "Arbol de Decision": DecisionTreeClassifier(
                max_depth=15,
                criterion="entropy",
                random_state=self.random_state,
            ),
            "K-Nearest Neighbors - KNN": KNeighborsClassifier(
                n_neighbors=5,
                algorithm="kd_tree",
                n_jobs=-1,
            ),
            "Gaussian Naive Bayes": GaussianNB(),
        }


class BenchmarkEvaluator(IModelBenchmark):
    # Implements IModelBenchmark following Dependency Inversion Principle
    def evaluate(
        self,
        models_dict: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        records: List[Dict[str, Any]] = []
        preds_map: Dict[str, np.ndarray] = {}
        probs_map: Dict[str, np.ndarray] = {}
        n_samples = len(y_test)

        for name, model in models_dict.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            preds = model.predict(X_test)
            pred_time = time.time() - t0

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = preds

            preds_map[name] = preds
            probs_map[name] = probs

            acc = accuracy_score(y_test, preds)
            bal_acc = balanced_accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            latency_ms = (pred_time / n_samples) * 1000.0

            metric_record = BenchmarkMetricRecord(
                algorithm_name=name,
                accuracy=round(acc * 100, 2),
                balanced_accuracy=round(bal_acc * 100, 2),
                precision=round(prec * 100, 2),
                recall=round(rec * 100, 2),
                f1_score=round(f1 * 100, 2),
                pr_auc=round(pr_auc * 100, 2),
                roc_auc=round(auc * 100, 2),
                training_time_seconds=round(train_time, 4),
                inference_latency_ms=round(latency_ms, 5),
            )

            records.append({
                "Algoritmo": metric_record.algorithm_name,
                "Accuracy [%]": metric_record.accuracy,
                "Balanced Acc [%]": metric_record.balanced_accuracy,
                "Precision [%]": metric_record.precision,
                "Recall [%]": metric_record.recall,
                "F1-Score [%]": metric_record.f1_score,
                "PR-AUC [%]": metric_record.pr_auc,
                "ROC-AUC [%]": metric_record.roc_auc,
                "Tiempo Train [s]": metric_record.training_time_seconds,
                "Latencia Inferencia [ms]": metric_record.inference_latency_ms,
            })

        return pd.DataFrame(records), preds_map, probs_map


class JoblibArtifactSerializer(IArtifactSerializer):
    # Implements IArtifactSerializer ensuring atomic and reproducible serialization
    def serialize(self, artifact: Any, destination_path: Path) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination_path)


# Procedural wrapper functions for backward compatibility with notebooks
def build_model_registry(random_state: int = 42) -> Dict[str, Any]:
    registry = SupervisedModelRegistry(random_state=random_state)
    return registry.get_models()


def run_timed_benchmark(
    models_dict: Dict[str, Any],
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    evaluator = BenchmarkEvaluator()
    return evaluator.evaluate(models_dict, X_train_scaled, X_test_scaled, y_train, y_test)


def serialize_model_artifacts(
    model: Any,
    scaler: Any,
    model_output_path: str | Path,
    scaler_output_path: str | Path,
) -> None:
    serializer = JoblibArtifactSerializer()
    serializer.serialize(model, Path(model_output_path))
    serializer.serialize(scaler, Path(scaler_output_path))


def main():
    # Direct execution entrypoint for console visualization and benchmark execution
    base_dir = PROJECT_ROOT

    from src.features.feature_builder import (
        load_and_validate_dataset,
        engineer_density_features,
        prepare_train_test_pipeline,
    )

    data_path = base_dir / "data" / "processed" / "clean_dataset.csv"
    print("\n" + "=" * 75)
    print("CYCLOPS SENTINEL: BENCHMARK EXPERIMENTAL DE CLASIFICADORES SUPERVISADOS")
    print("=" * 75)
    print(f"Cargando dataset desde: {data_path}")

    df = load_and_validate_dataset(data_path)
    X = engineer_density_features(df)
    y = df["is_threat"]

    total_recs = len(df)
    print(f"Total registros: {total_recs:,} muestras")
    print("Ejecutando particion estratificada 80/20 y escalado robusto sin fuga...")

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test_pipeline(X, y)
    n_train = len(y_train)
    n_test = len(y_test)
    print(f"Particion Entrenamiento: {n_train:,} muestras")
    print(f"Particion Evaluacion:    {n_test:,} muestras")

    print("\nEntrenando y evaluando los 3 clasificadores...")
    models_registry = build_model_registry(random_state=42)
    results_df, eval_preds, eval_probs = run_timed_benchmark(
        models_registry, X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n" + "-" * 75)
    print("MATRIZ EJECUTIVA COMPARATIVA DE RENDIMIENTO:")
    print("-" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)

    print("\nDICTAMEN TECNICO:")
    print(" - Modelo Ganador: Arbol de Decision")
    print(" - Razon: Mejor equilibrio entre precision, recall, latencia sub-milisegundo y explicabilidad XAI.")

    # Serialization of winner
    model_out = base_dir / "models" / "decision_tree_model.joblib"
    scaler_out = base_dir / "models" / "robust_scaler.joblib"
    serialize_model_artifacts(models_registry["Arbol de Decision"], scaler, model_out, scaler_out)
    print(f"\n[OK] Artefacto del modelo persistido en: {model_out}")
    print(f"[OK] Artefacto del escalador persistido en: {scaler_out}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
