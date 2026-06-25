"""
Sistema Integrato KB + ML + CSP
"""

import pandas as pd

import config
from src.intervention_planner import INTERVENTI


def _load_dataset_with_kb_features():
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    if config.KB_FEATURES_PATH.exists():
        kb_feats = pd.read_csv(config.KB_FEATURES_PATH)
        kb_cols = [c for c in kb_feats.columns if c.startswith("kb_")]
        if kb_cols:
            df = df.merge(kb_feats, on="patient_id", how="left")
            for col in kb_cols:
                df[col] = df[col].fillna(0).astype(int)
    return df


def _compute_kb_scores(sample_patients, kb):
    print(f"\n   [KB] Classificazione Rischio (regole Prolog ADA 2023)")
    print("-" * 55)
    scores = {}
    for _, row in sample_patients.iterrows():
        pid = int(row["patient_id"])
        risk = kb.classify_risk(pid)
        scores[pid] = config.RISK_TO_SCORE.get(risk, 0.0)
    return scores


def _compute_ml_scores(sample_patients, model, scaler, feature_cols, df):
    print(f"\n   [ML] Predizioni modello (probabilita' diabete)")
    print("-" * 55)
    scores = {}
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) != len(feature_cols):
        print(f"   ATTENZIONE: feature mancanti, ML scores a 0")
        return {int(row["patient_id"]): 0.0 for _, row in sample_patients.iterrows()}

    X_sample = sample_patients[feature_cols].values
    X_sample_scaled = scaler.transform(X_sample)
    ml_probs = model.predict_proba(X_sample_scaled)[:, 1]
    for i, (_, row) in enumerate(sample_patients.iterrows()):
        scores[int(row["patient_id"])] = float(ml_probs[i])
    return scores


def _compute_csp_scores(kb_scores, planner):
    print(f"\n   [CSP] Assegnamento piani di intervento")
    print("-" * 55)
    high_risk_pids = [pid for pid, score in kb_scores.items() if score >= 0.5]
    csp_scores = {pid: 0.0 for pid in kb_scores}

    if len(high_risk_pids) >= 2:
        try:
            solutions, _, info = planner.solve(
                patient_ids=high_risk_pids[:5], num_patients=min(5, len(high_risk_pids)),
                max_solutions=10)
            if solutions:
                for slot_name, inv_id in solutions[0].items():
                    pid = info[slot_name]["patient_id"]
                    csp_scores[pid] = 1.0
                    print(f"      Paziente {pid}: {INTERVENTI[inv_id]['nome']}")
        except Exception as e:
            print(f"   CSP integrazione: {e}")

    return csp_scores


def _adaptive_score(kb_s, ml_s, csp_s):
    if ml_s > config.ML_CONFIDENCE_HIGH or ml_s < config.ML_CONFIDENCE_LOW:
        return ml_s, True
    w = config.INTEGRATION_WEIGHTS
    return w["ml"] * ml_s + w["kb"] * kb_s + w["csp"] * csp_s, False


def run_integrated_demo(model, scaler, feature_cols, kb, planner, sample_size=20):
    """
    Calcola lo score integrato KB+ML+CSP per un campione di pazienti,
    stampa e salva il confronto con le valutazioni singole.
    """
    print("\n" + "=" * 60)
    print("   SISTEMA INTEGRATO: KB + ML + CSP")
    print("=" * 60)

    df = _load_dataset_with_kb_features()
    sample_patients = df.sample(n=min(sample_size, len(df)), random_state=config.RANDOM_STATE)

    kb_scores = _compute_kb_scores(sample_patients, kb)
    ml_scores = _compute_ml_scores(sample_patients, model, scaler, feature_cols, df)
    csp_scores = _compute_csp_scores(kb_scores, planner)

    print(f"\n   [Integrato] Score adattivo KB + ML + CSP")
    print(f"   Strategia: ML confidente (>0.6 o <0.4) -> usa solo ML")
    print(f"              ML incerto (0.4-0.6) -> 0.60*ML + 0.25*KB + 0.15*CSP")
    print("-" * 60)

    results = []
    n_confident = n_uncertain = 0
    for _, row in sample_patients.iterrows():
        pid = int(row["patient_id"])
        kb_s, ml_s, csp_s = kb_scores.get(pid, 0), ml_scores.get(pid, 0), csp_scores.get(pid, 0)
        score, confident = _adaptive_score(kb_s, ml_s, csp_s)
        n_confident += confident
        n_uncertain += not confident

        results.append({
            "patient_id": pid, "age": int(row["Age"]), "bmi": float(row["BMI"]),
            "actual": int(row["Outcome"]), "kb_score": kb_s, "ml_score": ml_s,
            "csp_score": csp_s, "integrated_score": score,
        })

    print(f"   ML confidente: {n_confident} pazienti | ML incerto (KB corregge): {n_uncertain} pazienti")
    results.sort(key=lambda x: x["integrated_score"], reverse=True)
    _print_results(results)
    _print_evaluation(results)
    _save_report(results)

    return results


def _print_results(results):
    for i, r in enumerate(results[:15], 1):
        sources = []
        if r["kb_score"] > 0:
            risk_label = ("critico" if r["kb_score"] >= 0.85 else
                          "alto" if r["kb_score"] >= 0.65 else
                          "moderato" if r["kb_score"] >= 0.35 else "basso")
            sources.append(f"KB:{risk_label}")
        if r["ml_score"] > 0:
            sources.append(f"ML:{r['ml_score']:.2f}")
        if r["csp_score"] > 0:
            sources.append("CSP:intervento")

        actual_label = "Diabete" if r["actual"] == 1 else "Sano"
        predicted_label = "Rischio" if r["integrated_score"] > 0.5 else "OK"
        print(f"      {i:2d}. Pz {r['patient_id']:3d} (Eta={r['age']:2d}, BMI={r['bmi']:.1f}) "
              f"Score:{r['integrated_score']:.4f}  Pred:{predicted_label:<8} "
              f"Reale:{actual_label:<8} [{', '.join(sources)}]")


def _print_evaluation(results):
    total = len(results)
    correct = sum(1 for r in results if (r["integrated_score"] > 0.5) == (r["actual"] == 1))
    kb_correct = sum(1 for r in results if (r["kb_score"] >= 0.5) == (r["actual"] == 1))
    ml_correct = sum(1 for r in results if (r["ml_score"] > 0.5) == (r["actual"] == 1))

    print(f"\n   Valutazione Sistema Integrato:")
    print(f"      KB sola:   {kb_correct}/{total} ({kb_correct / total * 100:.1f}%)")
    print(f"      ML sola:   {ml_correct}/{total} ({ml_correct / total * 100:.1f}%)")
    print(f"      Integrato: {correct}/{total} ({correct / total * 100:.1f}%)")


def _save_report(results):
    config.ensure_results_dir()
    total = len(results)
    correct = sum(1 for r in results if (r["integrated_score"] > 0.5) == (r["actual"] == 1))
    kb_correct = sum(1 for r in results if (r["kb_score"] >= 0.5) == (r["actual"] == 1))
    ml_correct = sum(1 for r in results if (r["ml_score"] > 0.5) == (r["actual"] == 1))

    with open(config.INTEGRATION_REPORT_PATH, "w") as f:
        f.write("CONFRONTO SISTEMA INTEGRATO - DIAGNOSI DIABETE TIPO 2\n")
        f.write("=" * 70 + "\n\n")
        f.write("Strategia integrazione adattiva:\n")
        f.write("  ML confidente (>0.6 o <0.4): score = ML\n")
        f.write("  ML incerto (0.4-0.6): score = 0.60*ML + 0.25*KB + 0.15*CSP\n\n")

        f.write(f"{'ID':>4} {'Eta':>4} {'BMI':>6} {'KB':>8} {'ML':>8} {'CSP':>8} "
                f"{'Integrato':>10} {'Predetto':>10} {'Reale':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            actual_label = "Diabete" if r["actual"] == 1 else "Sano"
            predicted_label = "Rischio" if r["integrated_score"] > 0.5 else "OK"
            f.write(f"{r['patient_id']:4d} {r['age']:4d} {r['bmi']:6.1f} "
                    f"{r['kb_score']:8.4f} {r['ml_score']:8.4f} {r['csp_score']:8.4f} "
                    f"{r['integrated_score']:10.4f} {predicted_label:>10} {actual_label:>10}\n")

        f.write(f"\nValutazione (campione {total} pazienti):\n")
        f.write(f"  KB sola:   {kb_correct}/{total} ({kb_correct / total * 100:.1f}%)\n")
        f.write(f"  ML sola:   {ml_correct}/{total} ({ml_correct / total * 100:.1f}%)\n")
        f.write(f"  Integrato: {correct}/{total} ({correct / total * 100:.1f}%)\n")

    print(f"\n   Confronto salvato in {config.INTEGRATION_REPORT_PATH.name}")
