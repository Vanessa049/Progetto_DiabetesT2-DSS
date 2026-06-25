"""
src - Package del Sistema di Supporto Decisionale per il Diabete di Tipo 2.

Moduli:
- data_processing:        EDA, gestione missing values, feature engineering ADA 2023
- reasoning:               interfaccia Python-Prolog, derivazione feature dalla KB
- supervised_learning:     addestramento e valutazione modelli ML (DT, RF, MLP)
- intervention_planner:    CSP per l'assegnazione di piani di intervento
- graph_search:            BFS vs A* su grafo di similarita' metabolica
- integration:             score integrato adattivo KB + ML + CSP
"""
