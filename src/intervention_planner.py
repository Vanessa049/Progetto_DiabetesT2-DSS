"""
CSP per Piani di Intervento Diabete T2

"""

import time

import pandas as pd
from constraint import AllDifferentConstraint, FunctionConstraint, Problem

import config

# ======================================================================
# CATALOGO INTERVENTI (conoscenza di dominio - linee guida ADA/EASD)
# ======================================================================

INTERVENTI = {
    "dieta_basso_ig": {
        "nome": "Dieta a Basso Indice Glicemico", "tipo": "lifestyle",
        "efficacia": 0.65, "costo": 0.10, "aderenza_stimata": 0.70,
        "bmi_min": 0, "eta_max": 100, "target": ["glicemia", "peso"],
    },
    "attivita_aerobica": {
        "nome": "Attivita' Fisica Aerobica (150 min/sett)", "tipo": "lifestyle",
        "efficacia": 0.70, "costo": 0.15, "aderenza_stimata": 0.60,
        "bmi_min": 0, "eta_max": 85, "target": ["peso", "insulino_resistenza", "glicemia"],
    },
    "metformina": {
        "nome": "Metformina (500-2000 mg/die)", "tipo": "farmaco",
        "efficacia": 0.80, "costo": 0.20, "aderenza_stimata": 0.75,
        "bmi_min": 0, "eta_max": 100, "target": ["glicemia", "insulino_resistenza"],
    },
    "cgm_monitoraggio": {
        "nome": "Monitoraggio Glicemico Continuo (CGM)", "tipo": "dispositivo",
        "efficacia": 0.60, "costo": 0.70, "aderenza_stimata": 0.85,
        "bmi_min": 0, "eta_max": 100, "target": ["glicemia", "compliance"],
    },
    "dsmes": {
        "nome": "Programma DSMES (Educazione Autogestione)", "tipo": "educazione",
        "efficacia": 0.65, "costo": 0.35, "aderenza_stimata": 0.65,
        "bmi_min": 0, "eta_max": 100, "target": ["compliance", "glicemia"],
    },
    "intervento_peso": {
        "nome": "Intervento Intensivo sul Peso (VLCD)", "tipo": "lifestyle_intensivo",
        "efficacia": 0.85, "costo": 0.50, "aderenza_stimata": 0.45,
        "bmi_min": 30, "eta_max": 75, "target": ["peso", "glicemia", "insulino_resistenza"],
    },
    "glp1_agonisti": {
        "nome": "GLP-1 Agonisti (Semaglutide)", "tipo": "farmaco",
        "efficacia": 0.88, "costo": 0.85, "aderenza_stimata": 0.70,
        "bmi_min": 27, "eta_max": 100, "target": ["glicemia", "peso", "cardiovascolare"],
    },
    "supporto_psicologico": {
        "nome": "Supporto Psicologico (CBT per aderenza)", "tipo": "psicosociale",
        "efficacia": 0.55, "costo": 0.40, "aderenza_stimata": 0.60,
        "bmi_min": 0, "eta_max": 100, "target": ["compliance", "stress"],
    },
}

# Coppie di interventi che non devono essere assegnate insieme nel gruppo
# (efficaci insieme ma richiedono supervisione stretta non garantita qui)
INTERAZIONI_INTERVENTI = [("metformina", "glp1_agonisti")]


class InterventionPlanner:
    """
    Risolutore CSP per l'assegnazione di piani di intervento a gruppi di
    pazienti a rischio diabete T2, con ottimizzazione multi-obiettivo
    post-soluzione (efficacia, aderenza, costo).
    """

    def __init__(self, patients_df: pd.DataFrame = None):
        self.patients_df = patients_df if patients_df is not None else self._load_data()

    @staticmethod
    def _load_data() -> pd.DataFrame:
        print("   Caricamento dati per CSP Solver...")
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        print(f"   {len(df)} pazienti caricati")
        print(f"   {len(INTERVENTI)} interventi disponibili: {list(INTERVENTI.keys())}")
        return df

    # ------------------------------------------------------------------
    # Costruzione del problema
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_interventions(patient_row):
        age = int(patient_row["Age"])
        bmi = float(patient_row["BMI"])
        return [inv_id for inv_id, inv in INTERVENTI.items()
                if age <= inv["eta_max"] and bmi >= inv["bmi_min"]]

    @staticmethod
    def _patient_needs(patient_row):
        needs = set()
        if patient_row.get("gluc_prediabete", 0) == 1 or patient_row.get("gluc_diabete", 0) == 1:
            needs.add("glicemia")
        if patient_row.get("bmi_obeso_i", 0) == 1 or patient_row.get("bmi_obeso_ii", 0) == 1:
            needs.update({"peso", "insulino_resistenza"})
        if patient_row.get("bp_ipertensione", 0) == 1:
            needs.add("pressione")
        if patient_row.get("insulina_alta", 0) == 1:
            needs.add("insulino_resistenza")
        return needs

    def solve(self, patient_ids=None, num_patients=5, max_solutions=50, timeout=30):
        """Risolve il CSP e restituisce (soluzioni, tempo_impiegato, info_pazienti)."""
        start_time = time.time()

        patients = self._select_patients(patient_ids, num_patients)
        print(f"\n   CSP: Assegnamento interventi per {len(patients)} pazienti")

        problem = Problem()
        slot_names, patient_info = [], {}

        for _, row in patients.iterrows():
            pid = int(row["patient_id"])
            slot_name = f"Paziente_{pid}"
            slot_names.append(slot_name)

            valid_invs = self._valid_interventions(row)
            patient_info[slot_name] = {
                "patient_id": pid, "age": int(row["Age"]), "bmi": float(row["BMI"]),
                "needs": self._patient_needs(row), "valid_interventions": valid_invs,
            }
            problem.addVariable(slot_name, valid_invs)

        if len(slot_names) < 2:
            print("   Troppo pochi pazienti per il CSP")
            return [], time.time() - start_time, patient_info

        problem.addConstraint(AllDifferentConstraint(), slot_names)
        self._add_no_interaction_constraints(problem, slot_names)

        solutions = self._collect_solutions(problem, max_solutions, timeout, start_time)
        elapsed = time.time() - start_time
        print(f"   CSP risolto: {len(solutions)} soluzioni trovate in {elapsed:.3f}s")
        return solutions, elapsed, patient_info

    def _select_patients(self, patient_ids, num_patients):
        if patient_ids is None:
            high_risk = self.patients_df[self.patients_df["Outcome"] == 1]
            if len(high_risk) < num_patients:
                high_risk = self.patients_df
            return high_risk.sample(n=min(num_patients, len(high_risk)),
                                    random_state=config.RANDOM_STATE)
        return self.patients_df[self.patients_df["patient_id"].isin(patient_ids)]

    @staticmethod
    def _add_no_interaction_constraints(problem, slot_names):
        def no_interaction(inv_a, inv_b):
            for a, b in INTERAZIONI_INTERVENTI:
                if {inv_a, inv_b} == {a, b}:
                    return False
            return True

        for i in range(len(slot_names)):
            for j in range(i + 1, len(slot_names)):
                problem.addConstraint(FunctionConstraint(no_interaction),
                                      [slot_names[i], slot_names[j]])

    @staticmethod
    def _collect_solutions(problem, max_solutions, timeout, start_time):
        solutions = []
        for sol in problem.getSolutionIter():
            solutions.append(sol)
            if len(solutions) >= max_solutions:
                break
            if time.time() - start_time > timeout:
                print(f"   CSP timeout ({timeout}s), {len(solutions)} soluzioni parziali")
                break
        return solutions

    # ------------------------------------------------------------------
    # Ottimizzazione (vincoli soft)
    # ------------------------------------------------------------------
    @staticmethod
    def score_solution(solution, patient_info, weights=None):
        """
        Punteggio pesato della soluzione: efficacia clinica rispetto ai
        bisogni del paziente, aderenza stimata, costo (da minimizzare).
        """
        weights = weights or config.CSP_WEIGHTS
        n = len(solution)
        total_efficacia = total_aderenza = total_costo = 0

        for slot_name, inv_id in solution.items():
            inv = INTERVENTI[inv_id]
            needs = patient_info[slot_name]["needs"]
            target_match = len(set(inv["target"]) & needs) / max(len(needs), 1)
            total_efficacia += inv["efficacia"] * (0.5 + 0.5 * target_match)
            total_aderenza += inv["aderenza_stimata"]
            total_costo += inv["costo"]

        return (weights["efficacia"] * (total_efficacia / n)
                + weights["aderenza"] * (total_aderenza / n)
                + weights["costo"] * (1 - total_costo / n))

    def solve_with_optimization(self, patient_ids=None, num_patients=5,
                                weights=None, max_solutions=50):
        """Risolve il CSP e ordina le soluzioni per punteggio decrescente."""
        print("\n   CSP con Ottimizzazione Multi-Obiettivo...")
        solutions, elapsed, patient_info = self.solve(
            patient_ids=patient_ids, num_patients=num_patients, max_solutions=max_solutions)

        if not solutions:
            print("   Nessuna soluzione CSP trovata")
            return [], elapsed, patient_info

        scored = sorted(
            ((sol, self.score_solution(sol, patient_info, weights)) for sol in solutions),
            key=lambda x: x[1], reverse=True)

        print(f"   Top score: {scored[0][1]:.4f}, Worst score: {scored[-1][1]:.4f}")
        return scored, elapsed, patient_info

    @staticmethod
    def solution_to_readable(solution, patient_info):
        """Converte una soluzione CSP in una lista di dizionari leggibili."""
        result = []
        for slot_name, inv_id in solution.items():
            info = patient_info[slot_name]
            inv = INTERVENTI[inv_id]
            result.append({
                "patient_id": info["patient_id"], "age": info["age"], "bmi": info["bmi"],
                "needs": info["needs"], "intervention_id": inv_id,
                "intervention_name": inv["nome"], "intervention_type": inv["tipo"],
                "efficacia": inv["efficacia"], "aderenza": inv["aderenza_stimata"],
                "costo": inv["costo"],
            })
        return result


def demo_intervention_planner():
    """Demo del CSP Solver con scenari rappresentativi di gruppi di pazienti a rischio."""
    print("=" * 60)
    print("   DEMO CSP - PIANI DI INTERVENTO DIABETE T2")
    print("=" * 60)

    planner = InterventionPlanner()
    scored, elapsed, patient_info = planner.solve_with_optimization(num_patients=6)

    config.ensure_results_dir()
    with open(config.INTERVENTION_REPORT_PATH, "w") as f:
        f.write("CSP - PIANI DI INTERVENTO DIABETE TIPO 2\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tempo risoluzione: {elapsed:.3f}s\n")
        f.write(f"Soluzioni trovate: {len(scored)}\n\n")

        if scored:
            best_solution, best_score = scored[0]
            f.write(f"MIGLIOR SOLUZIONE (score={best_score:.4f}):\n")
            f.write("-" * 60 + "\n")
            for item in planner.solution_to_readable(best_solution, patient_info):
                f.write(f"  Paziente {item['patient_id']} (Eta={item['age']}, "
                        f"BMI={item['bmi']:.1f}): {item['intervention_name']}\n")
                print(f"      Paziente {item['patient_id']}: {item['intervention_name']} "
                      f"(efficacia={item['efficacia']:.2f}, costo={item['costo']:.2f})")

    print(f"\n   Risultati salvati in {config.INTERVENTION_REPORT_PATH.name}")
    print("\n" + "=" * 60)
    print("   DEMO CSP COMPLETATA!")
    print("=" * 60)
    return planner


def main():
    return demo_intervention_planner()


if __name__ == "__main__":
    main()
