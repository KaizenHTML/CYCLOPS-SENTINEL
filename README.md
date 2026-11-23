# CYCLOPS: Mineria de Datos y Analitica Predictiva para Deteccion de Phishing

---

## 1. Descripcion del Proyecto

Este repositorio contiene la arquitectura especializada de **Mineria de Datos [Data Mining]**, Procesamiento de Lenguaje Natural [NLP] y Aprendizaje Automatico [Machine Learning] de CYCLOPS.

El objetivo central es transformar comunicaciones digitales no estructuradas [correos electronicos y mensajes SMS] en una representacion vectorial cuantitativa, permitiendo entrenar, evaluar y comparar algoritmos supervisados y no supervisados bajo el estandar metodologico **CRISP-DM / KDD**.

---

## 2. Estructura de Carpetas [Clean Data Science Architecture]

```text
CYCLOPS_DATA_MINING/
│
├── data/
│   ├── raw/                 # Datos inmutables originales [corporate_emails.csv, phishing_emails.csv, sms_spam.csv]
│   ├── interim/             # Datos intermedios unificados [unified_dataset.csv]
│   └── processed/           # Datos procesados con ingenieria de variables [normalized_dataset.csv, clean_dataset.csv]
│
├── notebooks/               # Cuadernos Jupyter reproducibles paso a paso
│   ├── 01_exploratory_data_analysis.ipynb       # Analisis exploratorio EDA, correlaciones y dispersiones
│   ├── 02_supervised_learning_benchmark.ipynb   # Benchmark: Arbol de Decision vs KNN vs Gaussian Naive Bayes
│   ├── 03_unsupervised_clustering.ipynb         # Segmentacion K-Means, metodo del codo y silueta
│   └── benchmark_clasificadores_supervisados.ipynb
│
├── src/                     # Codigo fuente modular y reutilizable
│   ├── data/                # Modulos del pipeline de datos
│   │   ├── ingest.py                    # Ingesta, balanceo y unificacion de esquemas
│   │   ├── normalize_and_anonymize.py   # Anonimizacion PII [URL, EMAIL, IP] y limpieza profunda
│   │   ├── feature_engineering.py       # Tokenizacion, stopwords, lematizacion y variables continuas
│   │   └── univariate_analysis.py       # Evaluacion de asimetria y distribuciones
│   │
│   └── models/              # Modulos de entrenamiento y validacion
│       └── evaluate_scalers.py          # Particion estratificada 80/20 y ajuste de RobustScaler
│
├── models/                  # Artefactos matematicos serializados [.joblib]
│   ├── decision_tree_model.joblib
│   ├── robust_scaler.joblib
│   ├── kmeans_model.joblib
│   └── kmeans_scaler.joblib
│
├── reports/                 # Evidencias analiticas
│   └── figures/             # Graficas de distribucion, matrices de confusion y curvas ROC-AUC
│
├── requirements.txt         # Dependencias estrictas de ciencia de datos
└── README.md                # Memoria tecnica del proyecto
```

---

## 3. Fases del Flujo de Mineria de Datos

### Fase 1: Ingesta y Balanceo
- Consolidacion de mas de 18 mil registros a traves de `src/data/ingest.py`.
- Homogenizacion de columnas hacia `text_content` y `is_threat`.

### Fase 2: Limpieza y Anonimizacion
- Ejecucion mediante `src/data/normalize_and_anonymize.py`.
- Reemplazo de PII mediante mascaras analiticas: `[URL]`, `[EMAIL]`, `[IP_ADDRESS]`, `[PHONE]`, `[CURRENCY]`.
- Supresion de codigo script, tags HTML y caracteres de control invisibles.

### Fase 3: Tokenizacion, Lematizacion y Extraccion de Caracteristicas
- Tokenizacion y reduccion morfológica mediante `WordNetLemmatizer` en `src/data/feature_engineering.py`.
- Extraccion de 8 parametros cuantitativos: `char_count`, `word_count`, `url_count`, `has_url`, `has_ip`, `urgency_score`, `url_density`, `urgency_density`.

### Fase 4: Escalado Robusto sin Fuga de Datos
- Particion 80/20 estratificada por clase objetivo.
- Ajuste de `RobustScaler` utilizando exclusivamente el 80% de entrenamiento con base en la mediana y el rango intercuartilico [IQR] para inmunizar el modelo ante valores atipicos [outliers].

### Fase 5: Modelado y Benchmark
- Evaluacion de tres familias de algoritmos en `notebooks/02_supervised_learning_benchmark.ipynb`:
  1. **Arbol de Decision:** Modelo ganador por su explicabilidad XAI y precision en particiones.
  2. **K-Nearest Neighbors [KNN]:** Clasificador basado en proximidad euclidiana con K=5.
  3. **Gaussian Naive Bayes:** Modelo probabilistico bayesiano continuo.
- Agrupamiento no supervisado K-Means en `notebooks/03_unsupervised_clustering.ipynb` con coeficiente de silueta.
