"""
Caricamento e Preprocessing dei Dati Clinici Diabete Tipo 2
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    Incapsula l'intera pipeline di preprocessing: caricamento, EDA,
    gestione missing values e feature engineering clinico.
    """

    NUMERIC_COLS_WITH_INVALID_ZEROS = ["Glucose", "BloodPressure", "BMI",
                                        "SkinThickness", "Insulin"]
    EDA_NUMERIC_COLS = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin",
                        "SkinThickness", "DiabetesPedigreeFunction"]

    def __init__(self, dataset_path: Path = None, results_dir: Path = None):
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.results_dir = results_dir or config.RESULTS_DIR
        self.df = None

    # ------------------------------------------------------------------
    # Caricamento
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Carica il file CSV del dataset Pima Indians Diabetes (UCI)."""
        print("   Caricamento dataset Pima Indians Diabetes...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"   Pazienti: {len(self.df)}")
        print(f"   Feature: {self.df.columns.tolist()}")
        return self.df

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def exploratory_data_analysis(self) -> None:
        """
        Analisi esplorativa: distribuzione target, statistiche cliniche,
        boxplot per classe e matrice di correlazione. Salva i grafici in results/figures/eda/.
        """
        print("\n   Exploratory Data Analysis...")
        config.ensure_results_dir()
        df = self.df

        self._plot_target_distribution(df)
        self._print_numeric_stats(df)
        self._plot_clinical_boxplots(df)
        self._plot_correlation_matrix(df)

        print(f"\n   Salvati grafici EDA in {config.RESULTS_FIGURES_EDA_DIR}")

    def _plot_target_distribution(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        target_counts = df["Outcome"].value_counts()
        axes[0].bar(["No Diabete (0)", "Diabete (1)"], target_counts.values,
                    color=["#2ecc71", "#e74c3c"], edgecolor="white")
        axes[0].set_ylabel("Frequenza")
        axes[0].set_title("Distribuzione Outcome Diabete")
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 3, f"{v} ({v / len(df) * 100:.1f}%)",
                         ha="center", fontweight="bold")

        df[df["Outcome"] == 0]["Age"].hist(bins=20, alpha=0.6, label="No Diabete",
                                           color="#2ecc71", ax=axes[1])
        df[df["Outcome"] == 1]["Age"].hist(bins=20, alpha=0.6, label="Diabete",
                                           color="#e74c3c", ax=axes[1])
        axes[1].set_xlabel("Eta'")
        axes[1].set_ylabel("Frequenza")
        axes[1].set_title("Distribuzione Eta' per Classe")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(config.EDA_TARGET_PLOT_PATH, dpi=150)
        plt.close()

    def _print_numeric_stats(self, df: pd.DataFrame) -> None:
        print("\n   Statistiche feature numeriche:")
        for col in self.EDA_NUMERIC_COLS:
            n_zero = (df[col] == 0).sum()
            print(f"      {col}: media={df[col].mean():.1f}, std={df[col].std():.1f}, "
                  f"min={df[col].min()}, max={df[col].max()}, zero={n_zero}")

        n_positive = (df["Outcome"] == 1).sum()
        n_negative = (df["Outcome"] == 0).sum()
        print("\n   Distribuzione target:")
        print(f"      Outcome=0 (no diabete): {n_negative} ({n_negative / len(df) * 100:.1f}%)")
        print(f"      Outcome=1 (diabete):    {n_positive} ({n_positive / len(df) * 100:.1f}%)")
        print(f"      Rapporto: {n_positive / n_negative:.2f}")

    def _plot_clinical_boxplots(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plots = [
            ("Glucose", "Glucosio per Classe", axes[0, 0]),
            ("BMI", "BMI per Classe", axes[0, 1]),
            ("Age", "Eta' per Classe", axes[0, 2]),
            ("Insulin", "Insulina per Classe", axes[1, 0]),
            ("BloodPressure", "Pressione Diastolica per Classe", axes[1, 1]),
            ("DiabetesPedigreeFunction", "Pedigree Diabetico per Classe", axes[1, 2]),
        ]
        for col, title, ax in plots:
            df.boxplot(column=col, by="Outcome", ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Outcome")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(config.EDA_CLINICAL_PLOT_PATH, dpi=150)
        plt.close()

    def _plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
        plt.title("Matrice di Correlazione - Dataset Diabete")
        plt.tight_layout()
        plt.savefig(config.EDA_CORRELATION_PLOT_PATH, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Imputa con mediana stratificata per classe i valori 0 biologicamente
        impossibili in Glucose, BloodPressure e BMI.
        """
        print("\n   Gestione valori mancanti...")
        df = self.df.copy()

        for col in self.NUMERIC_COLS_WITH_INVALID_ZEROS:
            df[col] = df[col].astype(float)

        impossible_zero_cols = {
            "Glucose": "glicemia zero non clinicamente possibile",
            "BloodPressure": "pressione diastolica zero impossibile",
            "BMI": "BMI zero impossibile",
        }

        for col, reason in impossible_zero_cols.items():
            n_zero = (df[col] == 0).sum()
            if n_zero > 0:
                print(f"   {col}=0 ({reason}): {n_zero} ({n_zero / len(df) * 100:.1f}%)")
                for cls in [0, 1]:
                    mask = (df[col] == 0) & (df["Outcome"] == cls)
                    median_val = df[(df[col] > 0) & (df["Outcome"] == cls)][col].median()
                    df.loc[mask, col] = median_val
                    print(f"      Classe {cls}: imputati con mediana = {median_val:.1f}")

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def feature_engineering(self) -> pd.DataFrame:
        """
        Feature engineering basato su soglie cliniche ADA 2023 / WHO / JNC 8:
        fasce eta', classi BMI, soglie glicemiche e pressorie, interazioni
        cliniche e conteggio fattori di rischio.
        """
        print("\n   Feature engineering clinico (linee guida ADA 2023)...")
        df = self.df.copy()

        # Fasce di eta' (rischio aumenta da 45 anni secondo ADA)
        df["age_giovane"] = (df["Age"] < 35).astype(int)
        df["age_medio"] = ((df["Age"] >= 35) & (df["Age"] < 45)).astype(int)
        df["age_rischio"] = ((df["Age"] >= 45) & (df["Age"] < 60)).astype(int)
        df["age_anziano"] = (df["Age"] >= 60).astype(int)

        # Classi BMI WHO
        df["bmi_normale"] = (df["BMI"] < 25).astype(int)
        df["bmi_sovrappeso"] = ((df["BMI"] >= 25) & (df["BMI"] < 30)).astype(int)
        df["bmi_obeso_i"] = ((df["BMI"] >= 30) & (df["BMI"] < 35)).astype(int)
        df["bmi_obeso_ii"] = (df["BMI"] >= 35).astype(int)

        # Soglie glicemiche ADA 2023
        df["gluc_normale"] = (df["Glucose"] < 100).astype(int)
        df["gluc_prediabete"] = ((df["Glucose"] >= 100) & (df["Glucose"] < 126)).astype(int)
        df["gluc_diabete"] = (df["Glucose"] >= 126).astype(int)

        # Soglie pressione diastolica (JNC 8)
        df["bp_normale"] = (df["BloodPressure"] < 80).astype(int)
        df["bp_alta"] = ((df["BloodPressure"] >= 80) & (df["BloodPressure"] < 90)).astype(int)
        df["bp_ipertensione"] = (df["BloodPressure"] >= 90).astype(int)

        # Insulina sierica (marker insulino-resistenza)
        df["insulina_alta"] = (df["Insulin"] > 130).astype(int)

        # Interazioni clinicamente rilevanti (normalizzate)
        df["bmi_x_gluc"] = df["BMI"] * df["Glucose"] / 1000
        df["age_x_bmi"] = df["Age"] * df["BMI"] / 1000
        df["age_x_gluc"] = df["Age"] * df["Glucose"] / 1000
        df["gluc_x_insulin"] = df["Glucose"] * df["Insulin"] / 10000

        # Conteggio fattori di rischio ADA 2023
        df["num_risk_factors"] = (
            (df["BMI"] >= 25).astype(int)
            + (df["Age"] >= 45).astype(int)
            + (df["Glucose"] >= 100).astype(int)
            + (df["BloodPressure"] >= 80).astype(int)
            + (df["DiabetesPedigreeFunction"] > 0.5).astype(int)
            + (df["Pregnancies"] >= 4).astype(int)
        )

        self.df = df
        print(f"   Feature engineering completato: {len(df.columns)} colonne totali")
        return df

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def build_training_dataset(self):
        """Aggiunge l'identificativo paziente e restituisce (df, feature_cols)."""
        print("\n   Creazione dataset di training...")

        df = self.df.copy()
        df["patient_id"] = range(1, len(df) + 1)
        feature_cols = [c for c in df.columns if c != "Outcome"]

        print(f"   Dataset: {len(df)} pazienti")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Distribuzione target: {df['Outcome'].value_counts().to_dict()}")

        self.df = df
        return df, feature_cols

    def save(self, df: pd.DataFrame, feature_cols: list) -> None:
        """Salva il dataset di training e l'elenco delle feature in results/data/."""
        print("\n   Salvataggio dati preprocessati...")
        config.ensure_results_dir()

        df.to_csv(config.TRAINING_DATA_PATH, index=False)
        with open(config.FEATURE_COLS_PATH, "w") as f:
            f.write("\n".join(feature_cols))

        print(f"   File salvati in {config.RESULTS_DATA_DIR}")

    # ------------------------------------------------------------------
    # Pipeline completa
    # ------------------------------------------------------------------
    def run(self):
        """Esegue l'intera pipeline di preprocessing e restituisce (df, feature_cols)."""
        self.load_data()
        self.exploratory_data_analysis()
        self.handle_missing_values()
        self.feature_engineering()
        df, feature_cols = self.build_training_dataset()
        self.save(df, feature_cols)
        return df, feature_cols


def main():
    print("=" * 60)
    print("   PREPROCESSING DATASET PIMA INDIANS DIABETES")
    print("=" * 60)

    processor = DataProcessor()
    df, feature_cols = processor.run()

    print("\n" + "=" * 60)
    print("   PREPROCESSING COMPLETATO!")
    print("=" * 60)

    return df, feature_cols


if __name__ == "__main__":
    main()
