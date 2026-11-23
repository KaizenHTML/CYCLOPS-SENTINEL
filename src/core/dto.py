from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelinePaths:
    # Immutable configuration for project data directories
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class BenchmarkMetricRecord:
    # Strongly typed data transfer object for evaluated model metrics
    algorithm_name: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    training_time_seconds: float
    inference_latency_ms: float
