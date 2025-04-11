import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from lingam import DirectLiNGAM

# === Load and clean dataset ===
csv_file = "uploaded_dataset.csv"
df = pd.read_csv(csv_file)

# Fix: Column names must be separated properly
df = df[["I_R01_Gripper_Load", "I_R01_Gripper_Pot", "I_R02_Gripper_Load"]]

# Drop rows with NaN or Inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Standardize
scaler = StandardScaler()
data = scaler.fit_transform(df.head(1000))

# === PATCH ICA STAGE TO AVOID DIVIDE BY ZERO ===
ica = FastICA()
S_ica = ica.fit_transform(data)
W_ica = ica.components_

# Patch: Avoid zero values in W_ica
W_safe = np.abs(W_ica)
W_safe[W_safe < 1e-8] = 1e-8  # Replace near-zero with small number
cost_matrix = 1 / W_safe
_, col_index = linear_sum_assignment(cost_matrix)
print("✅ ICA ordering:", col_index)

# === Run DirectLiNGAM ===
model = DirectLiNGAM()
model.fit(data)

# === Total effect: from variable 2 to 1 ===
from_var = 2  # I_R02_Gripper_Load
to_var = 1    # I_R01_Gripper_Pot

te = model.estimate_total_effect(data, from_var, to_var)
print(f"📊 Total effect of {df.columns[from_var]} → {df.columns[to_var]}: {te:.3f}")
