# DiabetesT2-DSS

Sistema intelligente di supporto decisionale per la diagnosi e la gestione del rischio del diabete di tipo 2, sviluppato per il corso di **Ingegneria della Conoscenza (ICon 25/26)**.

Il sistema integra una Knowledge Base Prolog, apprendimento automatico, CSP e ricerca euristica su grafo per classificare il rischio diabetico e pianificare interventi clinici.

**Autore:** Vanessa Antonia Dellaquila [802272], v.dellaquila3@studenti.uniba.it

## Struttura del progetto

```
DiabetesT2-DSS/
├── data/
│   └── diabetes.csv               # Pima Indians Diabetes Dataset (UCI, 768 pazienti)
├── docs/
│   ├── documentazione.docx        # Documentazione (Word)
│   └── documentazione.md          # Documentazione completa (sorgente)
├── knowledge_base/
│   └── diabetes_rules.pl          # Knowledge Base Prolog (regole ADA 2023)
├── results/                       # Output generati dalla pipeline, organizzati per tipo
│   ├── data/                      #   dataset intermedi e tabelle (CSV/TXT)
│   ├── models/                    #   modello migliore serializzato + scaler
│   ├── figures/
│   │   ├── eda/                   #   grafici dell'analisi esplorativa
│   │   ├── ml/                    #   grafici dell'apprendimento supervisionato
│   │   └── search/                #   grafico di confronto BFS vs A*
│   └── reports/                   #   report testuali di ciascuna fase
├── src/
│   ├── data_processing.py         # Caricamento dati, EDA, feature engineering
│   ├── reasoning.py               # Interfaccia Python-Prolog
│   ├── supervised_learning.py     # Classificazione con K-Fold CV e ablazione KB
│   ├── intervention_planner.py    # CSP per piani di intervento
│   ├── graph_search.py            # BFS vs A* su grafo di similarità metabolica
│   └── integration.py             # Score adattivo KB + ML + CSP
├── config.py                      # Percorsi e costanti condivise
├── main.py                        # Orchestratore principale (CLI)
└── requirements.txt                # Dipendenze Python
```

### Output della pipeline (`results/`)

Ogni fase scrive solo nella propria area, evitando di accumulare decine di file misti nella stessa cartella:

| Sottocartella | Contenuto | Generata da |
|---|---|---|
| `results/data/` | `dataset_processato.csv`, `elenco_feature.txt`, `feature_kb.csv`, `tabella_confronto_modelli.csv` | preprocessing, ragionamento KB, apprendimento |
| `results/models/` | `modello_migliore.pkl`, `scaler_dati.pkl` | apprendimento supervisionato |
| `results/figures/eda/` | `distribuzione_outcome.png`, `boxplot_cliniche.png`, `matrice_correlazione.png` | preprocessing (EDA) |
| `results/figures/ml/` | `importanza_feature.png`, `matrici_confusione.png`, `curve_roc.png`, `grafico_confronto_modelli.png` | apprendimento supervisionato |
| `results/figures/search/` | `grafico_bfs_astar.png` | ricerca su grafo |
| `results/reports/` | `report_ragionamento.txt`, `report_metriche_ml.txt`, `report_piani_intervento.txt`, `report_ricerca_grafo.txt`, `report_sistema_integrato.txt` | tutte le fasi |

Tutti i percorsi sono centralizzati in `config.py`: per cambiare la struttura delle cartelle di output basta modificare le costanti lì, senza toccare il codice dei singoli moduli.

## Requisiti

- Python 3.9+
- SWI-Prolog (richiesto solo per le fasi `--reason` e `--integrate`)

## Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SWI-Prolog va installato a livello di sistema operativo:

```bash
sudo apt install swi-prolog        # Ubuntu/Debian
brew install swi-prolog            # macOS
```

## Utilizzo

Per visualizzare tutti i comandi disponibili:

```bash
python3 main.py --help
```

### Pipeline completa

Esegue tutte le fasi in sequenza (preprocessing, ragionamento KB, apprendimento, CSP, ricerca su grafo, integrazione):

```bash
python3 main.py --all
```

### Fasi singole

Ogni fase importa solo i moduli di cui ha bisogno, quindi `--preprocess`, `--learn` e `--search` sono eseguibili anche senza SWI-Prolog/`python-constraint` installati:

```bash
python3 main.py --preprocess   # EDA, missing values, feature engineering
python3 main.py --reason       # Classificazione rischio + feature derivate dalla KB
python3 main.py --learn        # Training e valutazione modelli ML
python3 main.py --plan         # CSP per piani di intervento
python3 main.py --search       # BFS vs A* su grafo metabolico
python3 main.py --integrate    # Score adattivo KB + ML + CSP
```

## Moduli

| Modulo | Descrizione |
|--------|-------------|
| Preparazione dati | Caricamento CSV, missing values, feature engineering clinico (ADA 2023) |
| Knowledge Base Prolog | Classificazione rischio, diagnosi abduttiva, correlazioni metaboliche |
| Apprendimento Supervisionato | Decision Tree, Random Forest, MLP con K-Fold CV e studio di ablazione |
| CSP | Assegnamento piani di intervento con vincoli hard e ottimizzazione multi-obiettivo |
| Ricerca su grafo | Confronto BFS vs A* su grafo di similarità metabolica tra pazienti |
| Integrazione | Score adattivo KB + ML + CSP in base alla confidenza del modello |


