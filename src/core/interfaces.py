from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class IDataIngestor(ABC):
    # Single Responsibility: Ingesting and harmonizing raw source datasets
    @abstractmethod
    def ingest_and_harmonize(self, raw_data_dir: Path) -> pd.DataFrame:
        pass


class ITextSanitizer(ABC):
    # Single Responsibility: Cleaning text and masking sensitive personal identifiers
    @abstractmethod
    def sanitize_text(self, text: str) -> str:
        pass


class IFeatureTransformer(ABC):
    # Open Closed Principle: Open for new feature calculations, closed for core modifications
    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class IScalerPipeline(ABC):
    # Liskov Substitution Principle: Any scaler must respect fit and transform contract
    @abstractmethod
    def fit_and_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        pass


class IModelRegistry(ABC):
    # Interface Segregation: Specific catalog interface for machine learning models
    @abstractmethod
    def get_models(self) -> Dict[str, Any]:
        pass


class IModelBenchmark(ABC):
    # Dependency Inversion: High level evaluation depends on abstraction not concrete models
    @abstractmethod
    def evaluate(
        self,
        models_dict: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        pass


class IArtifactSerializer(ABC):
    # Single Responsibility: Persisting trained models and transformations
    @abstractmethod
    def serialize(self, artifact: Any, destination_path: Path) -> None:
        pass
