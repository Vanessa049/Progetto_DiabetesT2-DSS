"""
Interfaccia Python-Prolog per Ragionamento Metabolico
"""
import pandas as pd

import config

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except ImportError:
    PROLOG_AVAILABLE = False


class KnowledgeBaseReasoner:
    """
    Gestisce la Knowledge Base Prolog per la diagnosi del diabete di tipo 2:
    popolamento dei fatti, query di classificazione/diagnosi/correlazione,
    e derivazione delle feature per il modulo ML.
    """

    def __init__(self, kb_path=None):
        if not PROLOG_AVAILABLE:
            raise RuntimeError(
                "SWI-Prolog/pyswip non disponibile.\n"
                "Installazione richiesta:\n"
                "  1. sudo apt install swi-prolog   (Ubuntu/Debian)\n"
                "     oppure: sudo pacman -S swi-prolog  (Arch/Manjaro)\n"
                "  2. pip install pyswip"
            )

        self.kb_path = kb_path or config.KB_PROLOG_PATH
        self.df = None
        self.kb_features = None
        self.prolog = Prolog()
        self._load_kb()

    def _load_kb(self):
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge Base non trovata: {self.kb_path}")
        self.prolog.consult(str(self.kb_path))
        print("   Knowledge Base Prolog caricata")

    # ------------------------------------------------------------------
    # Popolamento KB
    # ------------------------------------------------------------------
    def load_patients(self, training_data_path=None):
        """Carica i pazienti preprocessati e popola la KB con i loro dati clinici."""
        print("   Caricamento dati per Knowledge Base...")
        path = training_data_path or config.TRAINING_DATA_PATH
        self.df = pd.read_csv(path)
        print(f"   {len(self.df)} pazienti caricati")
        self._populate_kb()

    def _populate_kb(self):
        """
        Asserisce in Prolog, per ogni paziente:
        - paziente(ID, Eta, Gravidanze, Pedigree, NumRiskFactors)
        - esame(PazID, TipoEsame, Valore) per ogni parametro clinico
        """
        print("   Popolamento KB Prolog con fatti clinici...")
        count = 0
        for _, row in self.df.iterrows():
            pid = int(row["patient_id"])
            try:
                self.prolog.assertz(
                    f"paziente({pid}, {int(row['Age'])}, {int(row['Pregnancies'])}, "
                    f"{float(row['DiabetesPedigreeFunction']):.4f}, {int(row['num_risk_factors'])})"
                )
                self.prolog.assertz(f"esame({pid}, glucose, {float(row['Glucose']):.1f})")
                self.prolog.assertz(f"esame({pid}, blood_pressure, {float(row['BloodPressure']):.1f})")
                self.prolog.assertz(f"esame({pid}, bmi, {float(row['BMI']):.1f})")
                self.prolog.assertz(f"esame({pid}, insulin, {float(row['Insulin']):.1f})")
                self.prolog.assertz(f"esame({pid}, skin_thickness, {float(row['SkinThickness']):.1f})")
                count += 1
            except Exception:
                pass
        print(f"   {count} pazienti aggiunti alla KB Prolog")

    def add_patient_fact(self, pid, age, pregnancies, pedigree, num_risk_factors,
                          glucose, blood_pressure, bmi, insulin, skin_thickness):
        """Asserisce manualmente i fatti per un paziente sintetico (es. scenari demo)."""
        self.prolog.assertz(f"paziente({pid}, {age}, {pregnancies}, {pedigree}, {num_risk_factors})")
        self.prolog.assertz(f"esame({pid}, glucose, {glucose})")
        self.prolog.assertz(f"esame({pid}, blood_pressure, {blood_pressure})")
        self.prolog.assertz(f"esame({pid}, bmi, {bmi})")
        self.prolog.assertz(f"esame({pid}, insulin, {insulin})")
        self.prolog.assertz(f"esame({pid}, skin_thickness, {skin_thickness})")

    # ------------------------------------------------------------------
    # Query di ragionamento
    # ------------------------------------------------------------------
    def classify_risk(self, patient_id) -> str:
        """Classifica il livello di rischio (basso/moderato/alto/critico) via backward chaining."""
        try:
            results = list(self.prolog.query(f"livello_rischio({patient_id}, Livello)"))
            if results:
                return str(results[0]["Livello"])
        except Exception as e:
            print(f"   Errore classificazione rischio: {e}")
        return "sconosciuto"

    def get_intervention_recommendation(self, patient_id) -> list:
        """Restituisce le raccomandazioni cliniche per il paziente."""
        try:
            results = list(self.prolog.query(f"raccomanda_intervento({patient_id}, Tipo)"))
            return [str(r["Tipo"]) for r in results]
        except Exception as e:
            print(f"   Errore intervento: {e}")
            return []

    def diagnose_patient(self, patient_id) -> list:
        """Diagnosi abduttiva: elenca i fattori metabolici che spiegano il rischio."""
        try:
            results = list(self.prolog.query(f"diagnosi_rischio({patient_id}, Causa)"))
            return [str(r["Causa"]) for r in results]
        except Exception as e:
            print(f"   Errore diagnosi: {e}")
            return []

    def check_metabolic_syndrome(self, patient_id) -> bool:
        """Verifica se il paziente soddisfa i criteri di sindrome metabolica."""
        try:
            return len(list(self.prolog.query(f"sindrome_metabolica({patient_id})"))) > 0
        except Exception:
            return False

    def check_correlations(self, factor) -> list:
        """Restituisce i fattori metabolici noti correlati a quello dato."""
        try:
            results = list(self.prolog.query(f"fattori_correlati({factor}, X)"))
            return [str(r["X"]) for r in results]
        except Exception as e:
            print(f"   Errore correlazioni: {e}")
            return []

    # ------------------------------------------------------------------
    # Derivazione feature per il ML
    # ------------------------------------------------------------------
    def derive_kb_features(self) -> pd.DataFrame:
        """
        Deriva, per ogni paziente, le 4 feature sintetizzate dalla KB:
        - kb_screening_level (0-3): livello screening raccomandato
        - kb_num_diagnosi (0-10): numero di fattori dalla diagnosi abduttiva
        - kb_fattori_correlati (0-6): coppie di fattori metabolici co-presenti
        - kb_profilo_tipico (0/1): pattern classico T2DM riconosciuto
        """
        print("   Derivazione feature dalla Knowledge Base...")

        features = []
        for _, row in self.df.iterrows():
            pid = int(row["patient_id"])
            feat = {"patient_id": pid}

            feat["kb_screening_level"] = self._query_scalar(
                f"is_screening_level({pid}, V)", default=0)
            feat["kb_num_diagnosi"] = self._query_scalar(
                f"is_num_diagnosi({pid}, V)", default=0)
            feat["kb_fattori_correlati"] = self._query_scalar(
                f"is_num_coppie_correlate({pid}, V)", default=0)
            feat["kb_profilo_tipico"] = self._query_scalar(
                f"is_profilo_tipico({pid}, V)", default=0)

            features.append(feat)

        self.kb_features = pd.DataFrame(features)
        self._print_kb_feature_stats()

        config.ensure_results_dir()
        self.kb_features.to_csv(config.KB_FEATURES_PATH, index=False)
        print(f"   Feature KB salvate in {config.KB_FEATURES_PATH.name}")

        return self.kb_features

    def _query_scalar(self, query, default=0):
        try:
            results = list(self.prolog.query(query))
            return int(results[0]["V"]) if results else default
        except Exception:
            return default

    def _print_kb_feature_stats(self):
        kb_cols = [c for c in self.kb_features.columns if c.startswith("kb_")]
        print(f"   {len(kb_cols)} feature derivate dalla KB:")
        for col in kb_cols:
            vals = self.kb_features[col]
            if vals.max() <= 1:
                n_positive = (vals == 1).sum()
                print(f"      {col}: {n_positive} pazienti positivi su {len(self.kb_features)}")
            else:
                print(f"      {col}: media={vals.mean():.2f}, min={vals.min()}, max={vals.max()}")


def demo_reasoning():
    """Demo del sistema di ragionamento con inferenza reale su un campione di pazienti."""
    print("=" * 60)
    print("   DEMO RAGIONAMENTO CON KNOWLEDGE BASE")
    print("=" * 60)

    kb = KnowledgeBaseReasoner()
    kb.load_patients()

    risk_counts = {"critico": 0, "alto": 0, "moderato": 0, "basso": 0}
    sample_results = []

    print("\n   Classificazione Rischio Diabetico (ADA 2023)")
    print("-" * 55)
    for pid in range(1, min(len(kb.df) + 1, 21)):
        risk = kb.classify_risk(pid)
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        row = kb.df[kb.df["patient_id"] == pid].iloc[0]
        sample_results.append({
            "patient_id": pid, "age": int(row["Age"]), "bmi": float(row["BMI"]),
            "glucose": float(row["Glucose"]), "risk": risk, "actual": int(row["Outcome"]),
        })
        if pid <= 10:
            print(f"      Paziente {pid:3d}: Eta={int(row['Age']):2d}, "
                  f"BMI={row['BMI']:.1f}, Gluc={row['Glucose']:.0f}, "
                  f"Rischio={risk:<10} | Reale={'Diabete' if row['Outcome'] == 1 else 'Sano'}")

    print("\n   Distribuzione rischio (primi 20 pazienti):")
    for level, count in risk_counts.items():
        print(f"      {level}: {count}")

    print("\n   Diagnosi Abduttiva (pazienti ad alto/critico rischio)")
    print("-" * 55)
    high_risk = [r for r in sample_results if r["risk"] in ("alto", "critico")]
    for patient in high_risk[:5]:
        pid = patient["patient_id"]
        diagnosi = kb.diagnose_patient(pid)
        print(f"      Paziente {pid} (Eta={patient['age']}, BMI={patient['bmi']:.1f}, "
              f"Gluc={patient['glucose']:.0f}):")
        for d in diagnosi[:6]:
            print(f"         -> {d}")

    print("\n   Raccomandazioni Intervento Clinico")
    print("-" * 55)
    for patient in sample_results[:8]:
        pid = patient["patient_id"]
        interventi = kb.get_intervention_recommendation(pid)
        if interventi:
            print(f"      Paziente {pid} (Rischio={patient['risk']}): "
                  f"{', '.join(interventi[:2])}")

    print("\n   Rilevamento Sindrome Metabolica")
    print("-" * 55)
    sm_count = sum(1 for p in sample_results if kb.check_metabolic_syndrome(p["patient_id"]))
    print(f"   Totale sindrome metabolica: {sm_count}/{len(sample_results)} pazienti")

    print("\n   Correlazioni tra Fattori di Rischio Metabolici")
    print("-" * 55)
    for factor in ["obesita", "insulino_resistenza", "eta_avanzata", "ipertensione"]:
        corr = kb.check_correlations(factor)
        if corr:
            print(f"      {factor} -> {', '.join(corr)}")

    print("\n   Derivazione Feature KB per ML")
    print("-" * 55)
    kb.derive_kb_features()

    _save_reasoning_report(kb, sample_results, risk_counts, high_risk)

    print("\n" + "=" * 60)
    print("   DEMO RAGIONAMENTO COMPLETATA!")
    print("=" * 60)
    return kb


def _save_reasoning_report(kb, sample_results, risk_counts, high_risk):
    config.ensure_results_dir()
    with open(config.REASONING_REPORT_PATH, "w") as f:
        f.write("RAGIONAMENTO CON KNOWLEDGE BASE PROLOG - DIABETE TIPO 2\n")
        f.write("=" * 60 + "\n\n")

        f.write("Classificazione Rischio (primi 20 pazienti):\n")
        for r in sample_results:
            f.write(f"  Paziente {r['patient_id']:3d}: Eta={r['age']:2d}, "
                    f"BMI={r['bmi']:.1f}, Gluc={r['glucose']:.0f}, "
                    f"Rischio={r['risk']:<10}, "
                    f"Reale={'Diabete' if r['actual'] == 1 else 'Sano'}\n")
        f.write("\nDistribuzione rischio:\n")
        for level, count in risk_counts.items():
            f.write(f"  {level}: {count}\n")

        f.write("\nDiagnosi Abduttiva (pazienti alto/critico rischio):\n")
        for patient in high_risk[:5]:
            pid = patient["patient_id"]
            f.write(f"  Paziente {pid} (Eta={patient['age']}, BMI={patient['bmi']:.1f}):\n")
            for d in kb.diagnose_patient(pid)[:6]:
                f.write(f"    -> {d}\n")

        f.write("\nFeature KB derivate:\n")
        kb_cols = [c for c in kb.kb_features.columns if c.startswith("kb_")]
        for col in kb_cols:
            vals = kb.kb_features[col]
            f.write(f"  {col}: media={vals.mean():.2f}, min={vals.min()}, max={vals.max()}\n")

    print(f"\n   Risultati salvati in {config.REASONING_REPORT_PATH.name}")


def main():
    return demo_reasoning()


if __name__ == "__main__":
    main()
