import pandas as pd
import numpy as np
from itertools import combinations
from causallearn.search.FCMBased.lingam import DirectLiNGAM

# Path to your uploaded dataset
csv_file = "uploaded_dataset.csv"
df = pd.read_csv(csv_file)

# Replace with your selected features
selected_features = ["I_R03_Gripper_Pot","I_R03_Gripper_Load", "I_R02_Gripper_Pot","I_R02_Gripper_Load", "I_R01_Gripper_Pot",
                     "I_R01_Gripper_Load", "I_R04_Gripper_Pot", "I_R04_Gripper_Load"
]

# Bootstrapping settings
n_bootstrap = 20
random_state = 42
edge_strengths = {}

# Run LiNGAM repeatedly on bootstrapped samples
for b in range(n_bootstrap):
    sample_df = df[selected_features].sample(frac=1.0, replace=True, random_state=random_state + b)
    model = DirectLiNGAM()
    model.fit(sample_df.values)

    adj_matrix = model.adjacency_matrix_
    labels = selected_features

    for i, j in combinations(range(len(labels)), 2):
        edge_strengths.setdefault((labels[i], labels[j]), []).append(adj_matrix[i, j])
        edge_strengths.setdefault((labels[j], labels[i]), []).append(adj_matrix[j, i])

# Collect statistics
results = []
for (src, tgt), vals in edge_strengths.items():
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    results.append({
        "Source": src,
        "Target": tgt,
        "Mean Strength": round(mean_val, 4),
        "Std Dev": round(std_val, 4),
        "Stability Score": round(1 / (1 + std_val), 4)  # Lower std = higher stability
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values("Stability Score", ascending=False))
