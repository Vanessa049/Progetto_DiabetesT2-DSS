"""
Configurazione centrale del progetto
"""

from pathlib import Path

# --- Percorsi base del progetto ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
KB_DIR = BASE_DIR / "knowledge_base"

DATASET_PATH = DATA_DIR / "diabetes.csv"
KB_PROLOG_PATH = KB_DIR / "diabetes_rules.pl"

# --- Sottocartelle di results/ ---
RESULTS_DATA_DIR = RESULTS_DIR / "data"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_FIGURES_EDA_DIR = RESULTS_FIGURES_DIR / "eda"
RESULTS_FIGURES_ML_DIR = RESULTS_FIGURES_DIR / "ml"
RESULTS_FIGURES_SEARCH_DIR = RESULTS_FIGURES_DIR / "search"
RESULTS_REPORTS_DIR = RESULTS_DIR / "reports"

# Elenco di tutte le sottocartelle che devono esistere prima di scrivere output
_RESULTS_SUBDIRS = (
    RESULTS_DATA_DIR,
    RESULTS_MODELS_DIR,
    RESULTS_FIGURES_EDA_DIR,
    RESULTS_FIGURES_ML_DIR,
    RESULTS_FIGURES_SEARCH_DIR,
    RESULTS_REPORTS_DIR,
)

# File intermedi prodotti dalla pipeline (usati per passare dati tra le fasi)
TRAINING_DATA_PATH = RESULTS_DATA_DIR / "dataset_processato.csv"
FEATURE_COLS_PATH = RESULTS_DATA_DIR / "elenco_feature.txt"
KB_FEATURES_PATH = RESULTS_DATA_DIR / "feature_kb.csv"
MODEL_COMPARISON_CSV_PATH = RESULTS_DATA_DIR / "tabella_confronto_modelli.csv"

# --- Modelli serializzati (src/supervised_learning.py) ---
BEST_MODEL_PATH = RESULTS_MODELS_DIR / "modello_migliore.pkl"
SCALER_PATH = RESULTS_MODELS_DIR / "scaler_dati.pkl"

# --- Output del preprocessing / EDA (src/data_processing.py) ---
EDA_TARGET_PLOT_PATH = RESULTS_FIGURES_EDA_DIR / "distribuzione_outcome.png"
EDA_CLINICAL_PLOT_PATH = RESULTS_FIGURES_EDA_DIR / "boxplot_cliniche.png"
EDA_CORRELATION_PLOT_PATH = RESULTS_FIGURES_EDA_DIR / "matrice_correlazione.png"

# --- Output del ragionamento KB (src/reasoning.py) ---
REASONING_REPORT_PATH = RESULTS_REPORTS_DIR / "report_ragionamento.txt"

# --- Output dell'apprendimento supervisionato (src/supervised_learning.py) ---
FEATURE_IMPORTANCE_PLOT_PATH = RESULTS_FIGURES_ML_DIR / "importanza_feature.png"
CONFUSION_MATRICES_PLOT_PATH = RESULTS_FIGURES_ML_DIR / "matrici_confusione.png"
ROC_CURVES_PLOT_PATH = RESULTS_FIGURES_ML_DIR / "curve_roc.png"
MODEL_COMPARISON_PLOT_PATH = RESULTS_FIGURES_ML_DIR / "grafico_confronto_modelli.png"
ML_METRICS_REPORT_PATH = RESULTS_REPORTS_DIR / "report_metriche_ml.txt"

# --- Output del CSP (src/intervention_planner.py) ---
INTERVENTION_REPORT_PATH = RESULTS_REPORTS_DIR / "report_piani_intervento.txt"

# --- Output della ricerca su grafo (src/graph_search.py) ---
GRAPH_SEARCH_PLOT_PATH = RESULTS_FIGURES_SEARCH_DIR / "grafico_bfs_astar.png"
GRAPH_SEARCH_REPORT_PATH = RESULTS_REPORTS_DIR / "report_ricerca_grafo.txt"

# --- Output dell'integrazione KB + ML + CSP (src/integration.py) ---
INTEGRATION_REPORT_PATH = RESULTS_REPORTS_DIR / "report_sistema_integrato.txt"

RANDOM_STATE = 42

# --- Pesi per lo score integrato KB + ML + CSP ---
INTEGRATION_WEIGHTS = {"ml": 0.60, "kb": 0.25, "csp": 0.15}
ML_CONFIDENCE_LOW = 0.4
ML_CONFIDENCE_HIGH = 0.6

# --- Pesi per l'ottimizzazione delle soluzioni CSP ---
CSP_WEIGHTS = {"efficacia": 0.40, "aderenza": 0.35, "costo": 0.25}

RISK_TO_SCORE = {"critico": 0.9, "alto": 0.7, "moderato": 0.4, "basso": 0.1}


def ensure_results_dir():
    """Crea la cartella results/ e tutte le sue sottocartelle, se non esistono."""
    RESULTS_DIR.mkdir(exist_ok=True)
    for subdir in _RESULTS_SUBDIRS:
        subdir.mkdir(parents=True, exist_ok=True)
