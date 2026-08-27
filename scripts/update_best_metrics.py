"""Atualiza best_metrics.json com base no metrics.csv recém-criado."""
import os
import json
import csv

EXP_DIR = "experiments/sphere_data_no_terrain_moved_terrain0/0"
CSV_PATH = os.path.join(EXP_DIR, "metrics.csv")
BEST_PATH = os.path.join(EXP_DIR, "best_metrics.json")

# Lê todos os rows do CSV
rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Lidas {len(rows)} linhas.")

# Coleta thresholds disponíveis
threshold_keys = [k for k in rows[0].keys() if k.startswith("f1_")]
threshold_keys.sort(key=lambda x: float(x.replace("f1_", "")))
print(f"Thresholds encontrados: {threshold_keys}")

# Critério: usa o threshold do meio (mesma lógica de update_best_metrics)
selected_key = threshold_keys[len(threshold_keys) // 2]
print(f"Threshold selecionado (do meio): {selected_key}")

# Encontra o melhor
best_row = None
for row in rows:
    val = float(row[selected_key])
    if best_row is None or val > float(best_row[selected_key]):
        best_row = row
        best_val = val

print(f"Melhor iteração: {best_row['iteration']} com {selected_key} = {best_val:.4f}")

best_data = {
    "best_iteration": int(best_row["iteration"]),
    "selection_metric": selected_key,
    "value": float(best_val),
    "mesh": best_row["mesh"],
    "metrics_file": f"metrics/{int(best_row['iteration']):08d}.json",
    "alignment": best_row.get("alignment", "centroid"),
    "all_f1": {k: float(best_row[k]) for k in threshold_keys},
    "accuracy": float(best_row["accuracy"]),
    "completeness": float(best_row["completeness"]),
    "chamfer_l1": float(best_row["chamfer_l1"]),
}

with open(BEST_PATH, "w") as f:
    json.dump(best_data, f, indent=2)

print(f"Salvo em {BEST_PATH}")