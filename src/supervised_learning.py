"""
supervised_learning.py - Apprendimento Supervisionato per Diagnosi Diabete Tipo 2

Responsabilita':
- 3 modelli di classificazione: Decision Tree, Random Forest, MLP
- Hyperparameter tuning con GridSearchCV
- Feature selection con SelectKBest (mutual information)
- Cross-validation stratificata 5-fold (media +/- deviazione standard)
- Studio di ablazione: confronto con/senza feature derivate dalla KB Prolog
- Salvataggio del modello migliore e dei grafici di confronto

Ingegneria della Conoscenza - UNIBA 25/26
"""

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config

warnings.filterwarnings("ignore")

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_models_and_grids():
    """
    Modelli e griglie per GridSearchCV.

    - Decision Tree: interpretabile, utile per estrarre regole cliniche
    - Random Forest: ensemble di alberi, riduce overfitting
    - MLP: approssimatore universale, cattura pattern non-lineari

    class_weight='balanced' gestisce lo sbilanciamento delle classi
    (65% no-diabete vs 35% diabete nel dataset PIMA).
    """
    return {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=config.RANDOM_STATE,
                                            class_weight="balanced"),
            "params": {"max_depth": [4, 8, 15], "min_samples_leaf": [3, 8],
                      "criterion": ["gini", "entropy"]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=config.RANDOM_STATE,
                                            class_weight="balanced", n_jobs=4),
            "params": {"n_estimators": [100, 200], "max_depth": [8, 15, None],
                      "min_samples_leaf": [2, 5]},
        },
        "MLP (Rete Neurale)": {
            "model": MLPClassifier(random_state=config.RANDOM_STATE, max_iter=500,
                                   early_stopping=True),
            "params": {"hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],
                      "alpha": [0.001, 0.01], "activation": ["relu", "tanh"]},
        },
    }


def _evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


def _cross_validate(X, y, model_class, params, n_splits=5):
    """5-fold stratified CV con scaling dentro ogni fold (no data leakage)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = {m: [] for m in METRICS}

    for train_idx, test_idx in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        model = model_class(**params)
        model.fit(X_train_scaled, y_train_fold)

        metrics, _, _ = _evaluate(model, X_test_scaled, y_test_fold)
        for m in METRICS:
            fold_metrics[m].append(metrics[m])

    return {
        "mean": {m: float(np.mean(fold_metrics[m])) for m in METRICS},
        "std": {m: float(np.std(fold_metrics[m])) for m in METRICS},
        "folds": fold_metrics,
    }


class SupervisedLearningPipeline:
    """
    Incapsula l'intera pipeline di apprendimento supervisionato: training con
    GridSearchCV, cross-validation, studio di ablazione KB, valutazione finale
    e salvataggio del modello migliore.
    """

    def __init__(self, results_dir=None, top_k_features=20):
        self.results_dir = results_dir or config.RESULTS_DIR
        self.top_k_features = top_k_features
        self.df = None
        self.feature_cols = None

    # ------------------------------------------------------------------
    # Caricamento dati (con integrazione feature KB)
    # ------------------------------------------------------------------
    def load_training_data(self):
        print("   Caricamento dati di training...")
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        with open(config.FEATURE_COLS_PATH, "r") as f:
            feature_cols = f.read().strip().split("\n")

        if config.KB_FEATURES_PATH.exists():
            print("   Integrazione feature derivate dalla KB Prolog...")
            kb_features = pd.read_csv(config.KB_FEATURES_PATH)
            kb_cols = [c for c in kb_features.columns if c.startswith("kb_")]
            if kb_cols:
                df = df.merge(kb_features, on="patient_id", how="left")
                for col in kb_cols:
                    df[col] = df[col].fillna(0).astype(int)
                feature_cols = feature_cols + kb_cols
                print(f"   {len(kb_cols)} feature KB integrate: {kb_cols}")
                with open(config.FEATURE_COLS_PATH, "w") as f:
                    f.write("\n".join(feature_cols))

        print(f"   {len(df)} esempi caricati, {len(feature_cols)} feature totali")
        self.df, self.feature_cols = df, feature_cols
        return df, feature_cols

    def prepare_data(self, test_size=0.2):
        print("\n   Preparazione dati...")
        X = self.df[self.feature_cols].values
        y = self.df["Outcome"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   Training set: {len(X_train)} esempi, Test set: {len(X_test)} esempi")
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def feature_selection(self, X_train, y_train):
        print(f"\n   Feature Selection (SelectKBest, k={self.top_k_features})...")

        k = min(self.top_k_features, len(self.feature_cols))
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_train, y_train)

        feature_scores = sorted(zip(self.feature_cols, selector.scores_),
                                 key=lambda x: x[1], reverse=True)

        print("   Top 10 feature per mutual information:")
        for feat, score in feature_scores[:10]:
            print(f"      {feat}: {score:.4f}")

        self._plot_feature_scores(feature_scores)

        selected_mask = selector.get_support(indices=False)
        selected_features = [f for f, s in zip(self.feature_cols, selected_mask) if s]
        return selector, selected_features, feature_scores

    def _plot_feature_scores(self, feature_scores):
        top = feature_scores[:20]
        plt.figure(figsize=(12, 8))
        names = [f[0] for f in top][::-1]
        values = [f[1] for f in top][::-1]
        plt.barh(range(len(names)), values, color="steelblue")
        plt.yticks(range(len(names)), names)
        plt.xlabel("Mutual Information Score")
        plt.title("Top Feature per Mutual Information - Diagnosi Diabete Tipo 2")
        plt.tight_layout()
        plt.savefig(config.FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Training, cross-validation, ablazione
    # ------------------------------------------------------------------
    def train_with_gridsearch(self, X_train, y_train, models_grids):
        print("\n   Hyperparameter Tuning con GridSearchCV...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        best_models = {}

        for name, cfg in models_grids.items():
            print(f"\n   [{name}] GridSearchCV in corso...")
            grid = GridSearchCV(estimator=cfg["model"], param_grid=cfg["params"],
                                cv=cv, scoring="f1", n_jobs=4, refit=True)
            grid.fit(X_train, y_train)

            best_models[name] = {"model": grid.best_estimator_,
                                  "best_params": grid.best_params_,
                                  "best_cv_score": grid.best_score_}
            print(f"   [{name}] Migliori parametri: {grid.best_params_}")
            print(f"   [{name}] Miglior F1 CV: {grid.best_score_:.4f}")

        return best_models

    def cross_validate_models(self, X, y, best_models, n_splits=5):
        print(f"\n   Valutazione con {n_splits}-Fold Stratified Cross-Validation...")
        cv_results = {}
        for name, info in best_models.items():
            print(f"\n   [{name}] Cross-validation {n_splits}-fold...")
            result = _cross_validate(X, y, info["model"].__class__,
                                      info["model"].get_params(), n_splits)
            cv_results[name] = result
            for m in METRICS:
                print(f"      {m:>10}: {result['mean'][m]:.4f} +/- {result['std'][m]:.4f}")
        return cv_results

    def ablation_kb_features(self, X, y, best_models, n_splits=5):
        """Confronta le performance con e senza le feature derivate dalla KB Prolog."""
        kb_indices = [i for i, c in enumerate(self.feature_cols) if c.startswith("kb_")]
        if not kb_indices:
            print("\n   Nessuna feature KB trovata, ablazione non necessaria.")
            return None

        kb_col_names = [self.feature_cols[i] for i in kb_indices]
        print(f"\n   Studio di Ablazione KB -> ML")
        print(f"   Feature KB rimosse: {kb_col_names}")

        non_kb_indices = [i for i, c in enumerate(self.feature_cols) if not c.startswith("kb_")]
        X_no_kb = X[:, non_kb_indices]

        ablation_results = {}
        for name, info in best_models.items():
            result = _cross_validate(X_no_kb, y, info["model"].__class__,
                                      info["model"].get_params(), n_splits)
            ablation_results[name] = result
            print(f"\n   [{name}] Senza feature KB (media +/- std):")
            for m in METRICS:
                print(f"      {m:>10}: {result['mean'][m]:.4f} +/- {result['std'][m]:.4f}")

        return ablation_results

    # ------------------------------------------------------------------
    # Valutazione
    # ------------------------------------------------------------------
    def evaluate_all_models(self, best_models, X_test, y_test):
        print("\n   Valutazione modelli sul test set...")
        all_results = {}
        for name, info in best_models.items():
            metrics, y_pred, y_prob = _evaluate(info["model"], X_test, y_test)
            all_results[name] = {"metrics": metrics, "y_pred": y_pred, "y_prob": y_prob,
                                  "model": info["model"], "best_params": info["best_params"],
                                  "best_cv_score": info["best_cv_score"]}
            print(f"\n   [{name}]")
            for m in METRICS:
                print(f"      {m}: {metrics[m]:.4f}")
        return all_results

    def evaluate_with_feature_selection(self, best_model_name, best_model, selector,
                                        X_train, X_test, y_train, y_test):
        print("\n   Confronto con/senza Feature Selection...")
        metrics_full, _, _ = _evaluate(best_model, X_test, y_test)

        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        model_sel = best_model.__class__(**best_model.get_params())
        model_sel.fit(X_train_sel, y_train)
        metrics_sel, _, _ = _evaluate(model_sel, X_test_sel, y_test)

        print(f"   [{best_model_name}] Tutte le feature: "
              f"F1={metrics_full['f1']:.4f}, AUC={metrics_full['roc_auc']:.4f}")
        print(f"   [{best_model_name}] Con feature selection (k={self.top_k_features}): "
              f"F1={metrics_sel['f1']:.4f}, AUC={metrics_sel['roc_auc']:.4f}")
        return metrics_full, metrics_sel

    # ------------------------------------------------------------------
    # Visualizzazioni
    # ------------------------------------------------------------------
    def plot_confusion_matrices(self, all_results, y_test):
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        for ax, (name, results) in zip(axes, all_results.items()):
            cm = confusion_matrix(y_test, results["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["No Diabete", "Diabete"],
                       yticklabels=["No Diabete", "Diabete"])
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("Reale")
            ax.set_xlabel("Predetto")
        plt.tight_layout()
        plt.savefig(config.CONFUSION_MATRICES_PLOT_PATH, dpi=150)
        plt.close()

    def plot_roc_curves(self, all_results, y_test):
        plt.figure(figsize=(10, 8))
        for name, results in all_results.items():
            fpr, tpr, _ = roc_curve(y_test, results["y_prob"])
            auc = results["metrics"]["roc_auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Confronto Curve ROC - Diagnosi Diabete Tipo 2")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.ROC_CURVES_PLOT_PATH, dpi=150)
        plt.close()

    def plot_models_comparison(self, all_results, cv_results):
        model_names = list(all_results.keys())
        comparison_data = []
        for name in model_names:
            row = {"Modello": name}
            for metric in METRICS:
                row[f"{metric}_mean"] = cv_results[name]["mean"][metric]
                row[f"{metric}_std"] = cv_results[name]["std"][metric]
            comparison_data.append(row)
        pd.DataFrame(comparison_data).to_csv(config.MODEL_COMPARISON_CSV_PATH, index=False)

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(model_names))
        width = 0.15
        for i, metric in enumerate(METRICS):
            values = [cv_results[name]["mean"][metric] for name in model_names]
            errors = [cv_results[name]["std"][metric] for name in model_names]
            ax.bar(x + i * width, values, width, yerr=errors, label=metric.upper(), capsize=3)

        ax.set_xlabel("Modello")
        ax.set_ylabel("Score")
        ax.set_title("Confronto Metriche - 5-Fold CV (media +/- std)")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.MODEL_COMPARISON_PLOT_PATH, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Salvataggio risultati
    # ------------------------------------------------------------------
    def save_results(self, all_results, feature_scores, cv_results, ablation_results):
        with open(config.ML_METRICS_REPORT_PATH, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("RISULTATI APPRENDIMENTO SUPERVISIONATO - DIAGNOSI DIABETE TIPO 2\n")
            f.write("=" * 70 + "\n\n")

            f.write("CONFRONTO MODELLI - 5-Fold Stratified Cross-Validation\n")
            f.write("-" * 90 + "\n")
            for name in cv_results:
                m, s = cv_results[name]["mean"], cv_results[name]["std"]
                f.write(f"{name:<25} " + " ".join(
                    f"{metric}={m[metric]:.4f}+/-{s[metric]:.4f}" for metric in METRICS) + "\n")

            if ablation_results:
                f.write("\n\nSTUDIO DI ABLAZIONE: CON vs SENZA FEATURE KB\n")
                f.write("-" * 90 + "\n")
                for name in cv_results:
                    f1_with = cv_results[name]["mean"]["f1"]
                    f1_without = ablation_results[name]["mean"]["f1"]
                    f.write(f"{name:<25} F1(con KB)={f1_with:.4f}  "
                            f"F1(senza KB)={f1_without:.4f}  Delta={f1_with - f1_without:+.4f}\n")

            f.write("\n\nMIGLIORI IPERPARAMETRI (GridSearchCV)\n")
            f.write("-" * 70 + "\n")
            for name, results in all_results.items():
                f.write(f"\n{name}:\n  Parametri: {results['best_params']}\n"
                        f"  Miglior F1 CV: {results['best_cv_score']:.4f}\n")

            if feature_scores:
                f.write("\n\nFEATURE RANKING (Mutual Information)\n")
                f.write("-" * 70 + "\n")
                for i, (feat, score) in enumerate(feature_scores[:30], 1):
                    f.write(f"  {i:2}. {feat:<40} {score:.4f}\n")

        print(f"\n   Risultati salvati in {config.ML_METRICS_REPORT_PATH.name}")

    # ------------------------------------------------------------------
    # Pipeline completa
    # ------------------------------------------------------------------
    def run(self):
        self.load_training_data()
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = \
            self.prepare_data()

        selector, _, feature_scores = self.feature_selection(X_train, y_train)

        models_grids = get_models_and_grids()
        best_models = self.train_with_gridsearch(X_train_scaled, y_train, models_grids)

        X_all = self.df[self.feature_cols].values
        y_all = self.df["Outcome"].values
        cv_results = self.cross_validate_models(X_all, y_all, best_models)
        ablation_results = self.ablation_kb_features(X_all, y_all, best_models)

        all_results = self.evaluate_all_models(best_models, X_test_scaled, y_test)

        best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean"]["f1"])
        print(f"\n   Modello migliore per F1 (CV): {best_model_name}")
        self.evaluate_with_feature_selection(
            best_model_name, all_results[best_model_name]["model"], selector,
            X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n   Generazione grafici...")
        self.plot_confusion_matrices(all_results, y_test)
        self.plot_roc_curves(all_results, y_test)
        self.plot_models_comparison(all_results, cv_results)
        self.save_results(all_results, feature_scores, cv_results, ablation_results)

        best_model = all_results[best_model_name]["model"]
        joblib.dump(best_model, config.BEST_MODEL_PATH)
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"\n   Modello migliore ({best_model_name}) salvato in {config.BEST_MODEL_PATH.name}")

        return best_model, scaler, self.feature_cols


def main():
    print("=" * 60)
    print("   APPRENDIMENTO SUPERVISIONATO")
    print("=" * 60)

    config.ensure_results_dir()
    pipeline = SupervisedLearningPipeline()
    best_model, scaler, feature_cols = pipeline.run()

    print("\n" + "=" * 60)
    print("   APPRENDIMENTO COMPLETATO!")
    print("=" * 60)
    return best_model, scaler, feature_cols


if __name__ == "__main__":
    main()
