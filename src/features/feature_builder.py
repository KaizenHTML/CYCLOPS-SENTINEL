import os
import sys
from pathlib import Path
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.interfaces import IFeatureTransformer, IScalerPipeline

BASE_FEATURES: List[str] = [
    "char_count",
    "word_count",
    "url_count",
    "has_url",
    "has_ip",
    "urgency_score",
]

ENGINEERED_FEATURES: List[str] = [
    "url_density",
    "urgency_density",
]

ALL_FEATURE_COLUMNS: List[str] = BASE_FEATURES + ENGINEERED_FEATURES


class DensityFeatureTransformer(IFeatureTransformer):
    # Implements IFeatureTransformer following Single Responsibility Principle
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[BASE_FEATURES].copy()
        word_safe = np.maximum(X["word_count"].values, 1)
        X["url_density"] = (X["url_count"].values / word_safe) * 100.0
        X["urgency_density"] = (X["urgency_score"].values / word_safe) * 100.0
        return X


class RobustScalerPipeline(IScalerPipeline):
    # Implements IScalerPipeline guaranteeing strict anti data leakage
    def __init__(self):
        self.scaler = RobustScaler()

    def fit_and_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, self.scaler


# Procedural wrapper functions for backward compatibility with notebooks
def load_and_validate_dataset(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    df = pd.read_csv(path)
    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("")
        
    for col in BASE_FEATURES + ["is_threat"]:
        if col not in df.columns:
            raise KeyError(f"Required column missing from dataset: {col}")
            
    return df


def engineer_density_features(df: pd.DataFrame) -> pd.DataFrame:
    transformer = DensityFeatureTransformer()
    return transformer.compute_features(df)


def prepare_train_test_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, RobustScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    pipeline = RobustScalerPipeline()
    X_train_scaled, X_test_scaled, scaler = pipeline.fit_and_transform(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
