"""
Sistema di Supporto Decisionale per il Diabete di Tipo 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config


def step_preprocess():
    print("\n" + "=" * 60)
    print("   FASE 1: PREPROCESSING")
    print("=" * 60)
    from src.data_processing import DataProcessor
    return DataProcessor().run()


def step_reason():
    print("\n" + "=" * 60)
    print("   FASE 2: RAGIONAMENTO CON KNOWLEDGE BASE")
    print("=" * 60)
    from src.reasoning import KnowledgeBaseReasoner
    kb = KnowledgeBaseReasoner()
    kb.load_patients()
    kb.derive_kb_features()
    return kb


def step_learn():
    print("\n" + "=" * 60)
    print("   FASE 3: APPRENDIMENTO SUPERVISIONATO")
    print("=" * 60)
    from src.supervised_learning import SupervisedLearningPipeline
    config.ensure_results_dir()
    return SupervisedLearningPipeline().run()


def step_plan():
    print("\n" + "=" * 60)
    print("   FASE 4a: CSP - PIANI DI INTERVENTO")
    print("=" * 60)
    from src.intervention_planner import demo_intervention_planner
    return demo_intervention_planner()


def step_search():
    print("\n" + "=" * 60)
    print("   FASE 4b: RICERCA SU GRAFO METABOLICO")
    print("=" * 60)
    import pandas as pd
    from src.graph_search import demo_graph_search
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    return demo_graph_search(df)


def step_integrate(model=None, scaler=None, feature_cols=None, kb=None, planner=None):
    print("\n" + "=" * 60)
    print("   FASE 5: SISTEMA INTEGRATO KB + ML + CSP")
    print("=" * 60)
    import joblib
    from src.integration import run_integrated_demo
    from src.intervention_planner import InterventionPlanner
    from src.reasoning import KnowledgeBaseReasoner

    if model is None or scaler is None:
        model = joblib.load(config.BEST_MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
    if feature_cols is None:
        with open(config.FEATURE_COLS_PATH, "r") as f:
            feature_cols = f.read().strip().split("\n")
    if kb is None:
        kb = KnowledgeBaseReasoner()
        kb.load_patients()
    if planner is None:
        planner = InterventionPlanner()

    return run_integrated_demo(model, scaler, feature_cols, kb, planner)


def run_all():
    print("=" * 60)
    print("   SISTEMA DI SUPPORTO DECISIONALE - DIABETE TIPO 2")
    print("   Progetto Ingegneria della Conoscenza - UNIBA 25/26")
    print("=" * 60)

    try:
        step_preprocess()
        kb = step_reason()
        model, scaler, feature_cols = step_learn()
        step_search()
        planner = step_plan()
        step_integrate(model, scaler, feature_cols, kb, planner)

        print("\n" + "=" * 60)
        print("   ESECUZIONE COMPLETATA CON SUCCESSO!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n   File non trovato: {e}")
        print("   Assicurati che diabetes.csv sia nella cartella 'data/'")
        sys.exit(1)
    except Exception as e:
        print(f"\n   Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema di Supporto Decisionale per il Diabete di Tipo 2 (KB + ML + CSP)"
    )
    parser.add_argument("--all", action="store_true", help="Esegue l'intera pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Solo preprocessing")
    parser.add_argument("--reason", action="store_true", help="Solo ragionamento KB Prolog")
    parser.add_argument("--learn", action="store_true", help="Solo apprendimento supervisionato")
    parser.add_argument("--plan", action="store_true", help="Solo CSP per piani di intervento")
    parser.add_argument("--search", action="store_true", help="Solo ricerca su grafo (BFS vs A*)")
    parser.add_argument("--integrate", action="store_true", help="Solo demo integrazione KB+ML+CSP")

    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.preprocess:
        step_preprocess()
    elif args.reason:
        step_reason()
    elif args.learn:
        step_learn()
    elif args.plan:
        step_plan()
    elif args.search:
        step_search()
    elif args.integrate:
        step_integrate()
    else:
        parser.print_help()
        print("\nEsempi:")
        print("  python main.py --all")
        print("  python main.py --preprocess")
        print("  python main.py --reason")
        print("  python main.py --learn")
        print("  python main.py --plan")
        print("  python main.py --search")
        print("  python main.py --integrate")


if __name__ == "__main__":
    main()
