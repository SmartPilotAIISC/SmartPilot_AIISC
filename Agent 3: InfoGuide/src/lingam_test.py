import numpy as np
from causallearn.search.FCMBased import lingam

# 1. Generate synthetic data for demonstration.
np.random.seed(1234)
n_samples = 500

# Suppose the true DAG is:
#   X0 -> X1,  X0 -> X2,  X1 -> X2.
# With direct effects: X1 = 0.8*X0 + noise, X2 = 0.5*X0 - 0.3*X1 + noise.

X0 = np.random.randn(n_samples)
noise_1 = 0.5 * np.random.randn(n_samples)
noise_2 = 0.5 * np.random.randn(n_samples)

X1 = 0.8 * X0 + noise_1
X2 = 0.5 * X0 - 0.3 * X1 + noise_2

X = np.column_stack([X0, X1, X2])  # shape: (n_samples, 3)

# 2. Fit ICALiNGAM
model = lingam.ICALiNGAM(random_state=0, max_iter=1000)
model.fit(X)

# 3. Extract the adjacency matrix (direct path coefficients).
B = model.adjacency_matrix_
print("Adjacency matrix (direct effects):\n", B)

import matplotlib.pyplot as plt
import networkx as nx

# 1. Create a directed graph from adjacency matrix B
G = nx.DiGraph()

# Add nodes (we'll label them 0, 1, 2 to match columns in X)
num_vars = B.shape[0]
G.add_nodes_from(range(num_vars))

# Add edges with weights
for i in range(num_vars):
    for j in range(num_vars):
        weight = B[i, j]
        # Put a small threshold so we don't draw spurious near-zero edges
        if abs(weight) > 1e-5:
            # Edge direction is j -> i if B[i, j] != 0
            G.add_edge(j, i, weight=round(weight, 2))

# 2. Position the nodes in some layout
pos = nx.spring_layout(G, seed=42)  # or any other layout you prefer

# 3. Draw the graph
nx.draw(G, pos, with_labels=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("DAG of Direct Effects (LiNGAM Adjacency)")
plt.show()

I = np.eye(num_vars)
total_effects = np.linalg.inv(I - B)

print("\nTotal Effects Matrix = (I - B)^(-1):\n", total_effects)
print("Entry (i, j) is the total effect of variable j on variable i.")

plt.imshow(total_effects, interpolation='none')
plt.colorbar()
plt.title("Total Effects Matrix")
plt.xlabel("Variable j")
plt.ylabel("Variable i")
plt.show()
