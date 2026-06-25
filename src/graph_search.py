"""
Ricerca su Grafo di Similarita' Metabolica tra Pazienti
"""

import heapq
import time
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SIMILARITY_FEATURES = ["Glucose", "BMI", "Age", "BloodPressure",
                        "Insulin", "DiabetesPedigreeFunction"]
EDGE_THRESHOLD = 0.12  # soglia di distanza: grafo sparso, percorsi multi-hop


class PatientGraphSearch:
    """
    Costruisce e interroga il grafo di similarita' metabolica tra pazienti.
    Peso dell'arco = distanza metabolica normalizzata tra i due pazienti.
    """

    def __init__(self, patients_df: pd.DataFrame, top_n: int = 200,
                 edge_threshold: float = EDGE_THRESHOLD):
        self.patients_df = patients_df
        self.edge_threshold = edge_threshold
        self.graph = defaultdict(list)
        self.node_info = {}
        self.node_ids = set()
        self.sim_features = [f for f in SIMILARITY_FEATURES if f in patients_df.columns]

        self._build_nodes(patients_df.head(top_n))
        self._build_edges()

    # ------------------------------------------------------------------
    # Costruzione del grafo
    # ------------------------------------------------------------------
    def _build_nodes(self, sample: pd.DataFrame):
        for _, row in sample.iterrows():
            pid = int(row["patient_id"])
            self.node_ids.add(pid)
            self.node_info[pid] = {
                "age": int(row["Age"]), "outcome": int(row["Outcome"]),
                "glucose": float(row["Glucose"]), "bmi": float(row["BMI"]),
                "bp": float(row["BloodPressure"]),
                "features": row[self.sim_features].values.astype(float),
            }

        all_features = np.array([self.node_info[pid]["features"] for pid in self.node_ids])
        self.feat_min = all_features.min(axis=0)
        self.feat_max = all_features.max(axis=0)
        self.feat_range = self.feat_max - self.feat_min
        self.feat_range[self.feat_range == 0] = 1

    def _build_edges(self):
        node_list = list(self.node_ids)
        edges_count = 0
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                dist = self._metabolic_distance(a, b)
                if dist < self.edge_threshold:
                    self.graph[a].append((b, dist))
                    self.graph[b].append((a, dist))
                    edges_count += 1
        print(f"   Grafo metabolico: {len(self.node_ids)} nodi, {edges_count} archi")

    def _metabolic_distance(self, pid_a, pid_b) -> float:
        feat_a = self.node_info[pid_a]["features"]
        feat_b = self.node_info[pid_b]["features"]
        norm_a = (feat_a - self.feat_min) / self.feat_range
        norm_b = (feat_b - self.feat_min) / self.feat_range
        return float(np.sqrt(np.mean((norm_a - norm_b) ** 2)))

    # ------------------------------------------------------------------
    # Componenti connesse
    # ------------------------------------------------------------------
    def get_connected_component(self, start_id) -> set:
        visited = {start_id}
        queue = deque([start_id])
        while queue:
            current = queue.popleft()
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def get_largest_component(self) -> set:
        remaining = set(self.node_ids)
        largest = set()
        while remaining:
            comp = self.get_connected_component(next(iter(remaining)))
            if len(comp) > len(largest):
                largest = comp
            remaining -= comp
        return largest

    # ------------------------------------------------------------------
    # Ricerca non informata: BFS
    # ------------------------------------------------------------------
    def bfs(self, start_id, goal_fn):
        """Ricerca in ampiezza: trova il cammino con minor numero di archi verso un goal."""
        if start_id not in self.node_ids:
            return None, 0

        visited = {start_id}
        queue = deque([(start_id, [start_id])])
        nodes_expanded = 0

        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1

            if current != start_id and goal_fn(current, self.node_info.get(current, {})):
                return path, nodes_expanded

            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None, nodes_expanded

    # ------------------------------------------------------------------
    # Ricerca informata: A*
    # ------------------------------------------------------------------
    def astar(self, start_id, goal_fn, target_info=None):
        """
        Ricerca informata con euristica metabolica composita (ammissibile):
        media pesata delle distanze normalizzate dal profilo target
        (glicemia 35%, BMI 30%, eta' 20%, outcome 15%).
        """
        if start_id not in self.node_ids:
            return None, 0, float("inf")

        target_info = target_info or {}
        target = {
            "glucose": target_info.get("glucose", 145),
            "bmi": target_info.get("bmi", 34),
            "age": target_info.get("age", 52),
            "outcome": target_info.get("outcome", 1),
        }

        def heuristic(node_id):
            info = self.node_info.get(node_id, {})
            gluc_diff = abs(info.get("glucose", 100) - target["glucose"]) / 200.0
            bmi_diff = abs(info.get("bmi", 25) - target["bmi"]) / 30.0
            age_diff = abs(info.get("age", 40) - target["age"]) / 60.0
            outcome_diff = 0.2 if info.get("outcome", 0) != target["outcome"] else 0
            return 0.35 * gluc_diff + 0.30 * bmi_diff + 0.20 * age_diff + 0.15 * outcome_diff

        counter = 0
        g_score = {start_id: 0}
        open_set = [(heuristic(start_id), counter, start_id, [start_id], 0)]
        visited = set()
        nodes_expanded = 0

        while open_set:
            _, _, current, path, g = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current != start_id and goal_fn(current, self.node_info.get(current, {})):
                return path, nodes_expanded, g

            for neighbor, weight in self.graph.get(current, []):
                if neighbor in visited:
                    continue
                new_g = g + weight
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    counter += 1
                    heapq.heappush(open_set, (new_g + heuristic(neighbor), counter,
                                              neighbor, path + [neighbor], new_g))

        return None, nodes_expanded, float("inf")


def _pick_source_patient(graph, component):
    """Sceglie come sorgente un paziente giovane e sano, con fallback progressivi."""
    candidates = [pid for pid in component
                  if graph.node_info[pid]["outcome"] == 0
                  and graph.node_info[pid]["age"] < 35
                  and graph.node_info[pid]["glucose"] < 100
                  and graph.node_info[pid]["bmi"] < 26]
    if not candidates:
        candidates = [pid for pid in component
                      if graph.node_info[pid]["outcome"] == 0 and graph.node_info[pid]["age"] < 40]
    if not candidates:
        candidates = [pid for pid in component if graph.node_info[pid]["outcome"] == 0]
    return candidates[0] if candidates else None


def demo_graph_search(patients_df: pd.DataFrame = None):
    """Demo: confronto BFS vs A* dal paziente sano piu' giovane a un profilo diabetico marcato."""
    print("=" * 60)
    print("   RICERCA SU GRAFO METABOLICO: BFS vs A*")
    print("=" * 60)

    if patients_df is None:
        patients_df = pd.read_csv(config.TRAINING_DATA_PATH)

    graph = PatientGraphSearch(patients_df, top_n=200)
    largest_comp = graph.get_largest_component()
    print(f"   Componente connessa piu' grande: {len(largest_comp)} nodi")

    start_id = _pick_source_patient(graph, largest_comp)
    if start_id is None:
        print("   Nessun paziente sano trovato come sorgente")
        return graph

    start_info = graph.node_info[start_id]
    print(f"\n   Sorgente: Paziente {start_id} (Eta={start_info['age']}, Sano, "
          f"Glucose={start_info['glucose']:.0f}, BMI={start_info['bmi']:.1f})")

    goals_strict = [pid for pid in largest_comp
                    if graph.node_info[pid]["outcome"] == 1
                    and graph.node_info[pid]["glucose"] >= 130
                    and graph.node_info[pid]["bmi"] >= 30 and pid != start_id]

    if goals_strict:
        def goal_fn(node_id, info):
            return info.get("outcome", 0) == 1 and info.get("glucose", 0) >= 130 and info.get("bmi", 0) >= 30
        goal_desc = "paziente diabetico con glucose>=130 e BMI>=30"
    else:
        def goal_fn(node_id, info):
            return info.get("outcome", 0) == 1 and info.get("glucose", 0) >= 120
        goal_desc = "paziente diabetico con glucose>=120"

    target_info = {"glucose": 145, "bmi": 34, "age": 52, "outcome": 1}
    print(f"\n   Obiettivo: {goal_desc}")

    print(f"\n   [BFS] Ricerca in ampiezza (non informata)...")
    t0 = time.time()
    bfs_path, bfs_expanded = graph.bfs(start_id, goal_fn)
    bfs_time = time.time() - t0
    _print_path("BFS", bfs_path, bfs_expanded, graph)

    print(f"\n   [A*] Ricerca informata (euristica metabolica)...")
    t0 = time.time()
    astar_path, astar_expanded, astar_cost = graph.astar(start_id, goal_fn, target_info)
    astar_time = time.time() - t0
    _print_path("A*", astar_path, astar_expanded, graph)

    _print_comparison(bfs_path, bfs_expanded, bfs_time, astar_path, astar_expanded, astar_time)
    _plot_comparison(bfs_path, bfs_expanded, bfs_time, astar_path, astar_expanded, astar_time)
    _save_report(graph, start_id, start_info, goal_desc, bfs_path, bfs_expanded, bfs_time,
                 astar_path, astar_expanded, astar_cost, astar_time)

    return graph


def _print_path(label, path, expanded, graph):
    if not path:
        print(f"   [{label}] Nessun percorso trovato ({expanded} nodi espansi)")
        return
    print(f"   [{label}] Percorso trovato ({len(path)} nodi, {expanded} espansi):")
    for i, pid in enumerate(path):
        info = graph.node_info[pid]
        print(f"      {i + 1}. Paziente {pid} (Eta={info['age']}, Gluc={info['glucose']:.0f}, "
              f"BMI={info['bmi']:.1f}, {'Diabete' if info['outcome'] == 1 else 'Sano'})")


def _print_comparison(bfs_path, bfs_expanded, bfs_time, astar_path, astar_expanded, astar_time):
    bfs_len = len(bfs_path) if bfs_path else 0
    astar_len = len(astar_path) if astar_path else 0
    print(f"\n   Confronto BFS vs A*")
    print("-" * 50)
    print(f"   {'Nodi espansi':<25} {bfs_expanded:>12d} {astar_expanded:>12d}")
    print(f"   {'Lunghezza percorso':<25} {bfs_len:>12d} {astar_len:>12d}")
    print(f"   {'Tempo (s)':<25} {bfs_time:>12.4f} {astar_time:>12.4f}")
    if bfs_expanded > 0 and astar_expanded > 0:
        riduzione = (1 - astar_expanded / bfs_expanded) * 100
        confronto_label = "meno" if riduzione > 0 else "piu'"
        print(f"\n   A* ha espanso {abs(riduzione):.1f}% {confronto_label} nodi rispetto a BFS")


def _plot_comparison(bfs_path, bfs_expanded, bfs_time, astar_path, astar_expanded, astar_time):
    config.ensure_results_dir()
    bfs_len = len(bfs_path) if bfs_path else 0
    astar_len = len(astar_path) if astar_path else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metodi = ["BFS\n(non informata)", "A*\n(informata)"]
    espansi = [bfs_expanded, astar_expanded]
    axes[0].bar(metodi, espansi, color=["#e74c3c", "#2ecc71"], edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Nodi Espansi")
    axes[0].set_title("Confronto Nodi Espansi: BFS vs A*\n(Grafo Metabolico Diabete)")
    for i, v in enumerate(espansi):
        axes[0].text(i, v + max(espansi) * 0.02, str(v), ha="center", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, [bfs_len, astar_len], width, label="Lungh. percorso", color="#3498db")
    twin = axes[1].twinx()
    twin.bar(x + width / 2, [bfs_time, astar_time], width, label="Tempo (s)", color="#f39c12", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metodi)
    axes[1].set_ylabel("Lunghezza Percorso")
    twin.set_ylabel("Tempo (s)")
    axes[1].legend(loc="upper left")
    twin.legend(loc="upper right")
    axes[1].set_title("Percorso e Tempo: BFS vs A*")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.GRAPH_SEARCH_PLOT_PATH, dpi=150)
    plt.close()
    print(f"\n   Salvato grafico {config.GRAPH_SEARCH_PLOT_PATH.name}")


def _save_report(graph, start_id, start_info, goal_desc, bfs_path, bfs_expanded, bfs_time,
                  astar_path, astar_expanded, astar_cost, astar_time):
    config.ensure_results_dir()
    with open(config.GRAPH_SEARCH_REPORT_PATH, "w") as f:
        f.write("RICERCA SU GRAFO METABOLICO: BFS vs A*\n")
        f.write("=" * 60 + "\n")
        f.write(f"Grafo: {len(graph.node_ids)} nodi, soglia similarita'={graph.edge_threshold}\n")
        f.write(f"Sorgente: Paziente {start_id} (Eta={start_info['age']}, Sano)\n")
        f.write(f"Obiettivo: {goal_desc}\n\n")

        for label, path, expanded in (("BFS", bfs_path, bfs_expanded), ("A*", astar_path, astar_expanded)):
            f.write(f"{label}:\n")
            if path:
                f.write(f"  Percorso ({len(path)} nodi, {expanded} espansi):\n")
                for i, pid in enumerate(path):
                    info = graph.node_info[pid]
                    f.write(f"    {i + 1}. Paziente {pid} (Eta={info['age']}, "
                            f"Gluc={info['glucose']:.0f}, BMI={info['bmi']:.1f}, "
                            f"{'Diabete' if info['outcome'] == 1 else 'Sano'})\n")
            else:
                f.write(f"  Nessun percorso trovato ({expanded} nodi espansi)\n")
            f.write("\n")

        f.write(f"Confronto:\n")
        f.write(f"  Nodi espansi: BFS={bfs_expanded}, A*={astar_expanded}\n")
        f.write(f"  Tempo (s): BFS={bfs_time:.4f}, A*={astar_time:.4f}\n")


def main():
    return demo_graph_search()


if __name__ == "__main__":
    main()
