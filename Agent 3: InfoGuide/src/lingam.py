import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
from causallearn.search.FCMBased import lingam
import os
import pickle

# ================
# Node Descriptions and Colors
# ================
NODE_DESCRIPTIONS = {
    "FIT101": ("Sensor", "Flowmeter; Measures inflow into raw water tank."),
    "LIT101": ("Actuator", "Level Transmitter; Raw water tank level."),
    "AIT201": ("Sensor", "Conductivity analyzer; Measures NaCl level."),
    "AIT202": ("Actuator", "pH analyzer; Measures HCl level."),
    "AIT203": ("Sensor", "ORP analyzer; Measures NaOCl level.")
}

NODE_COLORS = {
    "Sensor": "#1f77b4",   # Blue
    "Actuator": "#ff7f0e", # Orange
    "Unknown": "#d62728"   # Red
}

def plot_lingam_causal_graph(adj_matrix, node_labels, filename="lingam_causal_graph.html"):
    """Creates an interactive causal graph with Pyvis and embedded node legends."""
    G = nx.DiGraph()

    # Add nodes and edges
    for i in range(len(node_labels)):
        G.add_node(node_labels[i])

    # Note: B[i, j] is the effect of j -> i in LiNGAM,
    # but for clarity in the Pyvis graph, we often draw i -> j if B[i,j]!=0.
    # Adjust as needed based on your arrow convention!
    #
    # If the "model.adjacency_matrix_" is using B[i, j] meaning "X_j -> X_i",
    # you might want to add_edge(node_labels[j], node_labels[i]) instead.
    #
    # For the default causallearn LiNGAM:
    # B[i, j] ≠ 0 => X_j -> X_i
    # so we do add_edge(j, i).

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i, j] != 0:
                # j -> i
                G.add_edge(node_labels[j], node_labels[i])

    # Create Pyvis Network
    net = Network(height="800px", width="80%", directed=True, notebook=True)
    net.toggle_physics(False)

    # Add nodes with color and description
    for node in G.nodes:
        node_type, desc = NODE_DESCRIPTIONS.get(node, ("Unknown", "No description available."))
        node_color = NODE_COLORS.get(node_type, NODE_COLORS["Unknown"])
        net.add_node(node, label=node, title=desc, color=node_color, size=20, physics=False)

    # Add edges
    for edge in G.edges():
        src, dst = edge
        net.add_edge(src, dst, title=f"{src} → {dst}", color="gray")

    # Save basic graph first
    net.save_graph(filename)

    # Inject custom HTML for legends
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Legend for descriptions (Right)
    legend_html = """
    <div style="position: fixed; top: 50px; right: 20px; width: 300px; background-color: white;
                padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px gray;
                font-family: Arial, sans-serif; overflow-y: auto; max-height: 80vh;">
        <h4 style="margin: 0; padding-bottom: 10px;">Node Descriptions</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
    """
    for node, (_, desc) in NODE_DESCRIPTIONS.items():
        legend_html += f"<li><strong>{node}</strong>: {desc}</li>"

    legend_html += """
        </ul>
    </div>
    """

    # Legend for types (Left)
    node_type_legend = """
    <div style="position: fixed; top: 50px; left: 20px; width: 200px; background-color: white;
                padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px gray;
                font-family: Arial, sans-serif;">
        <h4 style="margin: 0; padding-bottom: 10px;">Node Types</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
    """
    for typ, color in NODE_COLORS.items():
        node_type_legend += f"""<li style="color: {color}; font-weight: bold;">● {typ}</li>"""

    node_type_legend += """
        </ul>
    </div>
    """

    # Append legends before </body>
    html_content = html_content.replace("</body>", legend_html + node_type_legend + "</body>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Graph saved as {filename}. Open in a browser to view.")


# =========================
# Load dataset
# =========================
csv_file = "uploaded_dataset.csv"
df = pd.read_csv(csv_file)
#df = df.drop(columns=['_time', 'Description', 'actual_state'])

data = df.head(1000).to_numpy()
node_labels = df.columns.tolist()

# =========================
# Run LiNGAM
# =========================
print("\n### Running LiNGAM for Causal Discovery ###")
model = lingam.ICALiNGAM()
model.fit(data)

# =========================
# Extract adjacency matrix
# =========================
adj_matrix = model.adjacency_matrix_
print("Adjacency matrix (direct effects) B:\n", adj_matrix)

# =========================
# Plot interactive DAG (direct effects)
# =========================
plot_lingam_causal_graph(adj_matrix, node_labels, filename="lingam_causal_graph.html")

# =========================
# Build a networkx DiGraph for edges
# =========================
G = nx.DiGraph()
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix[i])):
        if adj_matrix[i, j] != 0:
            # j -> i in LiNGAM
            G.add_edge(node_labels[j], node_labels[i])

# Save edges for later usage
with open("lingam_graph_edges.pkl", "wb") as f:
    pickle.dump(list(G.edges()), f)

# =========================
# Compute TOTAL EFFECTS
# =========================
#
# For a linear SEM X = B X + e,
# the total effect of X_j on X_i
# is the (i,j)-th entry of (I - B)^(-1).
#

# 1. Identity matrix
I = np.eye(adj_matrix.shape[0])

# 2. Invert (I - B)
inv_matrix = np.linalg.inv(I - adj_matrix)

# 3. This inv_matrix is your total effects matrix.
#    Entry (i,j) => total effect of variable j on variable i (direct + indirect).
print("Total Effects Matrix ( (I - B)^(-1) ):\n", inv_matrix)

# Save total effects and adjacency with labels
with open("lingam_total_effects.pkl", "wb") as f:
    pickle.dump((inv_matrix, node_labels), f)

# Save adjacency matrix for loading in Streamlit
with open("lingam_adjacency_matrix.pkl", "wb") as f:
    pickle.dump((adj_matrix, node_labels), f)

print("\nAll artifacts saved:\n"
      " - lingam_causal_graph.html (interactive Pyvis DAG of direct effects)\n"
      " - lingam_graph_edges.pkl (pickled edge list of direct effects)\n"
      " - lingam_adjacency_matrix.pkl (saving if needed)\n"
      " - lingam_total_effects.pkl (the total effects matrix and column names)\n")
