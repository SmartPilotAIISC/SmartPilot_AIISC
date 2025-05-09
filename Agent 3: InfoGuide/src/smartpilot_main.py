import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM
import pandas as pd
import shutil
import uuid
from rdflib import Graph, Namespace, URIRef, RDF, RDFS
import subprocess
import glob
from sentence_transformers import SentenceTransformer, util
import pickle
import json
import numpy as np
from copilots.RootCause import parse_sensor_ranges, analyze_all, compute_rca_statistics
from copilots.ProcessOntologyQA import ProcessOntologyQA
#from assets.OntologyQuery import extract_neo4j_data
import os
import re

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

@st.cache_data
def load_lingam_matrix():
    try:
        with open("lingam_adjacency_matrix.pkl", "rb") as f:
            adj_matrix, node_labels = pickle.load(f)
        return adj_matrix, node_labels
    except Exception:
        return None, None

adj_matrix, node_labels = load_lingam_matrix()

@st.cache_data
def load_lingam_total_effects():
    try:
        with open("lingam_total_effects.pkl", "rb") as f:
            total_effects, node_labels = pickle.load(f)
        return total_effects, node_labels
    except Exception:
        return None, None

total_effects, node_labels_te = load_lingam_total_effects()


@st.cache_resource
def load_knowledge_graph():
    g = Graph()
    g.parse("assets/Analog24HrRunKG_Demo_New.ttl", format="turtle")
    return g

# Namespaces
ns = {
    "sm": Namespace("http://purl.org/net/SmartManufacturing/v00/"),
    "sosa": Namespace("http://www.w3.org/ns/sosa/"),
    "cora": Namespace("http://purl.org/ieee1872-owl/cora-bare#"),
    "dul": Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"),
    "om": Namespace("http://www.ontology-of-units-of-measure.org/resource/om-2/"),
    "rparts": Namespace("http://purl.org/ieee1872-owl/rParts/")
}

if "ProcessOntologyQa" not in st.session_state:
    # st.session_state["ProcessOntologyQa"] = ProcessOntologyQA("/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Agent 3: InfoGuide/src/assets/d3_graph.json")
    # Get the current directory where the script is located
    script_dir = os.path.dirname(__file__)

    # Define the relative path dynamically
    d3_graph_path = os.path.join(script_dir, "assets", "d3_graph.json")

    # Assign to session state
    st.session_state["ProcessOntologyQa"] = ProcessOntologyQA(d3_graph_path)
    
def get_full_entity_semantic_info(entity_name: str, ontology_json: dict):

    #nodes, edges = extract_neo4j_data()

    #d3_data = {
      #  "nodes": list(nodes.values()),  # Convert dict to list for D3.js
       # "links": edges
    #}

    nodes = ontology_json.get("nodes", [])
    links = ontology_json.get("links", [])

    entity_info = []
    entity_node = None
    name_to_node = {n["item_name"].lower(): n for n in nodes if "item_name" in n}
    id_to_node = {n["id"]: n for n in nodes}

    # Step 1: Match the exact node
    for node in nodes:
        if node.get("item_name", "").lower() == entity_name.lower():
            entity_node = node
            break

    if not entity_node:
        return [f"❌ No entity named `{entity_name}` found in the ontology."]

    entity_info.append(f"🆔 Name: {entity_node.get('item_name')}")
    if "type" in entity_node:
        entity_info.append(f"🔖 Type: {entity_node['type']}")
    if "description" in entity_node:
        entity_info.append(f"📝 Description: {entity_node['description']}")
    if "unit" in entity_node:
        entity_info.append(f"📏 Unit: {entity_node['unit']}")
    if "item_spec" in entity_node:
        entity_info.append(f"🏭 Manufacturer Info (from spec): {entity_node['item_spec']}")
    if entity_node.get("type", "").lower() == "sensor_value":
        entity_info.append(
            f"📏 Tolerance Range: {entity_node.get('min_value')} to {entity_node.get('max_value')} "
            f"(Unit: {entity_node.get('unit', 'N/A')})"
        )

    # Step 2: Look at related links
    related_info = []
    for link in links:
        rel = link.get("relationship", "related_to")
        if link.get("source") == entity_node.get("id"):
            target_node = id_to_node.get(link["target"])
            if target_node:
                target_name = target_node.get("item_name", "Unknown")
                related_info.append(f"➡️ {rel} → {target_name}")
                if "item_spec" in target_node:
                    entity_info.append(f"🏭 Related Manufacturer ({target_name}): {target_node['item_spec']}")
        elif link.get("target") == entity_node.get("id"):
            source_node = id_to_node.get(link["source"])
            if source_node:
                source_name = source_node.get("item_name", "Unknown")
                related_info.append(f"⬅️ {rel} ← {source_name}")
                if "item_spec" in source_node:
                    entity_info.append(f"🏭 Related Manufacturer ({source_name}): {source_node['item_spec']}")

    if related_info:
        entity_info.append("🔗 Relationships:")
        entity_info.extend([f"   - {rel}" for rel in related_info])

    return entity_info

def extract_entity_name(user_input: str, ontology_nodes: list):
    candidates = [n.get("item_name", "").lower() for n in ontology_nodes if "item_name" in n]
    for cand in candidates:
        if cand in user_input.lower():
            return cand
    # fallback: try regex for last "word number" pattern
    match = re.search(r"([A-Za-z_ ]+\d+)", user_input)
    if match:
        return match.group(1).strip().lower()
    return None


def detect_robot_sensor_query(user_query):
    # Extract robot number from user query
    match = re.search(r"robot\s*(\d+)", user_query.lower())
    if not match:
        return None, None  # No robot number mentioned

    robot_number = match.group(1)

    # Check semantic similarity to confirm intent
    sensor_robot_templates = [
        "what sensors are connected to robot",
        "which sensors are used in robot",
        "robot sensor list",
        "show sensors for robot",
        "sensors used by robot",
        "what are the sensors in robot"
    ]

    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    template_embeddings = embedder.encode(sensor_robot_templates, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, template_embeddings)[0]
    best_score = scores.max().item()

    if best_score > 0.7:  # Confirm it's a sensor_robot intent
        return robot_number, best_score
    return None, None

def detect_anomaly_type_query_semantic(user_query: str, threshold: float = 0.7) -> bool:
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)

    ANOMALY_QUESTION_TEMPLATES = [
        "what are the types of anomalies",
        "list anomaly types",
        "show anomaly classes",
        "what anomalies are defined in the knowledge graph",
        "which anomalies exist",
        "give anomaly categories",
        "list all kinds of anomalies",
    ]

    template_embeddings = embedder.encode(ANOMALY_QUESTION_TEMPLATES, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, template_embeddings)[0]
    best_score = scores.max().item()

    return best_score > threshold

def get_anomaly_types_from_kg(graph: Graph) -> list:
    base_uri = "http://purl.org/net/SmartManufacturing/v00/"
    anomaly_uri = URIRef(base_uri + "Anamoly")

    anomaly_types = []

    for subj, pred, obj in graph.triples((None, RDFS.subClassOf, anomaly_uri)):
        if isinstance(subj, URIRef):
            anomaly_name = subj.split("/")[-1]
            anomaly_types.append(anomaly_name)

    return anomaly_types


def get_feature_semantic_info(variable_name: str, graph: Graph):
    base_data_uri = "http://purl.org/net/SmartManufacturing/v00/data/"
    base_type_uri = "http://purl.org/net/SmartManufacturing/v00/"
    results = []

    # Fuzzy match based on known sensor type suffix
    sensor_type_map = {
        "Gripper_Load": "GripperLoadSensor",
        "Gripper_Pot": "GripperPotentiometerSensor",
        "Safety_Door": "SafetyDoor",
        "Stopper": "Stopper",
        "Temp": "Temperature",
        "Angle": "Angle",
        "HMI": "HMIEStopButton",
    }

    matched_type = None
    for keyword, sensor_class in sensor_type_map.items():
        if keyword.lower() in variable_name.lower():
            matched_type = sensor_class
            break

    if not matched_type:
        return ["No matchable sensor type for this variable."]

    # Find all subjects in KG that have this rdf:type
    for subj, _, obj in graph.triples((None, RDF.type, URIRef(base_type_uri + matched_type))):
        # For that subject, try to extract type/class/subclass/description
        description = []

        for _, _, comment in graph.triples((subj, RDFS.comment, None)):
            description.append(f"📝 {comment}")

        description.append(f"🔹 Type: {matched_type}")

        for _, _, superclass in graph.triples((subj, RDFS.subClassOf, None)):
            description.append(f"🔸 Subclass of: {superclass.split('/')[-1]}")

        if description:
            return description

    return ["No matching instance found for this sensor type."]

def get_full_feature_semantic_info(variable_name: str, graph: Graph):
    from rdflib.namespace import RDF, RDFS
    base_data_uri = "http://purl.org/net/SmartManufacturing/v00/data/"
    base_type_uri = "http://purl.org/net/SmartManufacturing/v00/"
    om = Namespace("http://www.ontology-of-units-of-measure.org/resource/om-2/")

    results = []

    # Fuzzy match based on known sensor type suffix
    sensor_type_map = {
        "Gripper_Load": "GripperLoadSensor",
        "Gripper_Pot": "GripperPotentiometerSensor",
        "Safety_Door": "SafetyDoor",
        "Stopper": "Stopper",
        "Temp": "Temperature",
        "Angle": "Angle",
        "HMI": "HMIEStopButton",
    }

    matched_type = None
    for keyword, sensor_class in sensor_type_map.items():
        if keyword.lower() in variable_name.lower():
            matched_type = sensor_class
            break

    if not matched_type:
        return ["No matchable sensor type for this variable."]

    for subj, _, obj in graph.triples((None, RDF.type, URIRef(base_type_uri + matched_type))):
        description = []

        # rdfs:comment
        for _, _, comment in graph.triples((subj, RDFS.comment, None)):
            description.append(f"📝 {comment}")

        description.append(f"🔹 Type: {matched_type}")

        # rdfs:subClassOf chain
        for _, _, superclass in graph.triples((subj, RDFS.subClassOf, None)):
            description.append(f"🔸 Subclass of: {superclass.split('/')[-1]}")

        # om:hasUnit
        for _, _, unit in graph.triples((subj, om.hasUnit, None)):
            description.append(f"📏 Unit of measure: {unit.split('/')[-1]}")

        # cora:robotPart / rparts:robotSensingPart
        for pred in [URIRef("http://purl.org/ieee1872-owl/cora-bare#robotPart"), URIRef("http://purl.org/ieee1872-owl/rParts/robotSensingPart")]:
            for _, _, part in graph.triples((subj, pred, None)):
                description.append(f"🔧 Has Part: {part.split('/')[-1]}")

        for pred in [URIRef("http://purl.org/ieee1872-owl/cora-bare#isPartOf"), URIRef("http://purl.org/ieee1872-owl/rParts/isPartOf")]:
            for _, _, part in graph.triples((subj, pred, None)):
                description.append(f"🧩 Is Part Of: {part.split('/')[-1]}")

        # sosa:observes
        for _, _, observed in graph.triples((subj, URIRef("http://www.w3.org/ns/sosa/observes"), None)):
            description.append(f"📡 Observes: {observed.split('/')[-1]}")

        # sosa:detects
        for _, _, detected in graph.triples((subj, URIRef("http://www.w3.org/ns/sosa/detects"), None)):
            description.append(f"🚨 Detects: {detected.split('/')[-1]}")

        if description:
            return description

    return ["No matching instance found for this sensor type."]


# Build Robot ID → UUID mapping and reverse
def build_robot_sensor_map(graph):
    robot_label_to_uuid = {}
    uuid_to_robot_label = {}
    robot_counter = 1

    for s, p, o in graph.triples((None, RDF.type, ns["cora"].Robot)):
        robot_uuid = s.split("/")[-1]
        label = f"Robot {robot_counter}"
        robot_label_to_uuid[label] = robot_uuid
        uuid_to_robot_label[robot_uuid] = label
        robot_counter += 1

    return robot_label_to_uuid, uuid_to_robot_label

# Load KG and build mappings
rdf_graph = load_knowledge_graph()
robot_label_to_uuid, uuid_to_robot_label = build_robot_sensor_map(rdf_graph)

# Store in session
st.session_state["robot_label_to_uuid"] = robot_label_to_uuid
st.session_state["uuid_to_robot_label"] = uuid_to_robot_label

def get_sensors_connected_to_robot(robot_number: str, graph: Graph, robot_map: dict):
    label = f"Robot {robot_number.zfill(1)}"
    robot_uuid = robot_map.get(label)

    if not robot_uuid:
        return f"❌ Robot {robot_number} not found in the knowledge graph."

    robot_uri = URIRef(f"http://purl.org/net/SmartManufacturing/v00/data/{robot_uuid}")
    sensor_info = []

    for _, _, sensor_uri in graph.triples((robot_uri, ns["rparts"].robotSensingPart, None)):
        sensor_uuid = sensor_uri.split("/")[-1]
        sensor_type = graph.value(subject=sensor_uri, predicate=RDF.type)
        sensor_type_name = sensor_type.split("/")[-1] if sensor_type else "UnknownSensor"
        sensor_info.append(f"{sensor_uuid} ({sensor_type_name})")

    if not sensor_info:
        return f"ℹ️ No sensors connected to Robot {robot_number}."

    return f"🔌 Sensors connected to Robot {robot_number}:\n\n- " + "\n- ".join(sensor_info)

def handle_causal_reasoning_query(user_query: str, adj_matrix, node_labels):
    original_query = user_query  # preserve original case
    lowered_query = user_query.lower()

    if adj_matrix is None or node_labels is None:
        return "⚠️ Causal graph not available. Please run LiNGAM first."

    # 1. Causal Strength Query
    match = re.search(r"strength.*between\s+([A-Za-z0-9_]+)\s+and\s+([A-Za-z0-9_]+)", original_query)
    if match:
        a, b = match.groups()

        # 🐛 DEBUGGING BLOCK
        st.write("🔍 Causal Strength Query:")
        st.write(f"🔹 Variable A: `{a}`")
        st.write(f"🔹 Variable B: `{b}`")
        st.write(f"📜 node_labels: {node_labels}")

        if a in node_labels and b in node_labels:
            i, j = node_labels.index(a), node_labels.index(b)
            st.write(f"📈 Total effects matrix value: total_effects[{i}, {j}] = {total_effects[i, j]}")
            strength = total_effects[i, j]  # Use total_effects instead of adj_matrix
            return f"🔗 The total causal strength from `{a}` → `{b}` is **{strength}**"

    # 2. Strongest Cause Query
    match = re.search(r"strongest.*(cause|effect).*on\s+([A-Za-z0-9_]+)", original_query)
    if match:
        _, target = match.groups()
        if target in node_labels:
            j = node_labels.index(target)
            causes = total_effects[:, j]  # Use total_effects here
            if all(v == 0 for v in causes):
                return f"ℹ️ No variable found with a causal effect on `{target}`."
            max_idx = int(np.argmax(np.abs(causes)))
            cause_var = node_labels[max_idx]
            strength = causes[max_idx]
            return f"🔥 `{cause_var}` has the strongest total causal effect on `{target}` (strength = {strength})"

    # 3. Interventional Query: If A were set to x, what is the effect on B?
    match = re.search(
        r'if\s+([a-zA-Z0-9_]+)\s+(?:were|was)\s+set\s+to\s+([-\d.]+).+?effect\s+on\s+([a-zA-Z0-9_]+)',
        original_query,
        re.IGNORECASE
    )
    if match:
        a, x, b = match.groups()
        if a in node_labels and b in node_labels:
            i, j = node_labels.index(a), node_labels.index(b)
            x = float(x)
            strength = total_effects[i, j]  # Use total_effects for intervention effect
            if strength == 0:
                return f"ℹ️ `{a}` does **not** cause `{b}` (total effect ≈ 0)."
            effect = x * strength
            return f"📉 If `{a}` were set to {x}, it would cause `{b}` to change approximately by **{effect:.4f}** units (via total causal strength {strength:.4f})."

    # 4. Query for direct causal parents (incoming edges to B)
    match = re.search(
        r'(?:what\s+directly\s+causes|show\s+causal\s+parents\s+of|what\s+are\s+the\s+causal\s+parents\s+of)\s+([A-Za-z0-9_]+)',
        lowered_query)
    if match:
        target_raw = match.group(1).strip().strip("?")  # Remove trailing question mark if any

        # Case-insensitive lookup
        matched_targets = [label for label in node_labels if label.lower() == target_raw.lower()]
        if matched_targets:
            target = matched_targets[0]
            j = node_labels.index(target)
            parents = [(node_labels[i], adj_matrix[i, j]) for i in range(len(node_labels)) if adj_matrix[i, j] != 0]
            if not parents:
                return f"ℹ️ No variable has a direct causal effect on `{target}`."
            parent_lines = [f"- `{p}` (strength = {s:.4f})" for p, s in parents]
            return f"🔍 Direct causal parents of `{target}`:\n\n" + "\n".join(parent_lines)
        else:
            return f"❌ Variable `{target_raw}` not found in the causal graph."

    return None

def answer_root_cause_query(query: str, rca_results: list):
    query = query.strip()
    cause_freq, avg_strengths, total_counts = compute_rca_statistics(rca_results)

    def normalize(name):
        return name.strip().lower()

    # Q1. Most likely root cause of B
    match = re.search(r"most likely root cause of .*variable ([\w\d_]+)", query, re.IGNORECASE)
    if match:
        b = normalize(match.group(1))
        b_causes = {k: v for k, v in avg_strengths.items() if normalize(k[1]) == b}
        if not b_causes:
            return f"⚠️ No root cause data found for `{b}`."
        top_cause = max(b_causes.items(), key=lambda x: x[1])
        return f"🔍 Most likely root cause of `{top_cause[0][1]}` is `{top_cause[0][0]}` with average strength {top_cause[1]:.4f}."

    # Q2. Top 3 root causes of B
    match = re.search(r"3 most likely root causes of .*variable ([\w\d_]+)", query, re.IGNORECASE)
    if match:
        b = normalize(match.group(1))
        b_causes = {k: v for k, v in avg_strengths.items() if normalize(k[1]) == b}
        if not b_causes:
            return f"⚠️ No root cause data found for `{b}`."
        top_3 = sorted(b_causes.items(), key=lambda x: -x[1])[:3]
        lines = [f"{i+1}. `{k[0]}` (avg strength: {v:.4f})" for i, (k, v) in enumerate(top_3)]
        return f"📊 Top 3 likely root causes of `{b}`:\n\n" + "\n".join(lines)

    # Q3. Is A a likely root cause of B?
    match = re.search(r"is ([\w\d_]+) a likely root cause of .*variable ([\w\d_]+)", query, re.IGNORECASE)
    if match:
        a, b = normalize(match.group(1)), normalize(match.group(2))
        key = (a, b)
        total = total_counts.get(b, 0)
        freq = cause_freq.get(key, 0)
        if total == 0:
            return f"⚠️ No data for `{b}`."
        ratio = freq / total
        if ratio >= 0.1:
            return f"✅ Yes, `{a}` is a likely root cause of `{b}` (present in {ratio*100:.1f}% of cases)."
        else:
            return f"❌ No, `{a}` is not a likely root cause of `{b}` (only in {ratio*100:.1f}% of cases)."

    # Q4: Which is more likely to be the root cause of variable X: A or B?
    match = re.search(
        r"which.*more likely.*root cause.*of (?:the )?variable ([\w\d_]+).*?(?:is it|,)?\s*([\w\d_]+)\s*(?:or|,)\s*([\w\d_]+)",
        query, re.IGNORECASE)
    if match:
        target_var = match.group(1).strip().lower()
        a = match.group(2).strip().lower()
        b = match.group(3).strip().lower()

        a_strength = avg_strengths.get((a, target_var), 0)
        b_strength = avg_strengths.get((b, target_var), 0)

        if a_strength == 0 and b_strength == 0:
            return f"⚠️ Neither `{a}` nor `{b}` is a known root cause of `{target_var}`."

        more_likely = a if a_strength > b_strength else b
        return (
            f"🔍 `{more_likely}` is more likely the root cause of `{target_var}` "
            f"(strengths: `{a}`={a_strength:.4f}, `{b}`={b_strength:.4f})."
        )

    # Q5. Why is D not a likely root cause of B?
    match = re.search(r"why is ([\w\d_]+) not.*?root cause of .*variable ([\w\d_]+)", query, re.IGNORECASE)
    if match:
        d, b = normalize(match.group(1)), normalize(match.group(2))
        key = (d, b)
        freq = cause_freq.get(key, 0)
        avg = avg_strengths.get(key, 0.0)
        if freq == 0:
            return f"🧐 `{d}` is not considered a likely root cause of `{b}` because it never appeared as a cause in the RCA results."
        else:
            return f"🧐 `{d}` is not considered a strong root cause of `{b}` because its average causal strength is low ({avg:.4f}) and it appeared in only {freq} cases."

    return "🤖 Sorry, I couldn't interpret that RCA query. Try rephrasing it or check for typos."


# Initialize session state keys before usage
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Ensuring messages is initialized as an empty list

if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = ""  # Ensure it's initialized as an empty string

if "selected_features" not in st.session_state or st.session_state["selected_features"] is None:
    st.session_state["selected_features"] = ""  # Ensure it's always a string

# Load models
def load_anomaly_prediction_model():
    model_checkpoint = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "final_best_model_PredictX")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    df = pd.read_excel('./LLM_FT_dataset.csv')
    unique_labels = df['predicted_label'].unique().tolist()
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    return tokenizer, model, id2label


def get_anomaly_prediction(tokenizer, model, id2label, user_query, time_series_data):
    new_text_inputs = [f"{series} {user_query}" for series in time_series_data]
    tokenized_inputs = tokenizer(new_text_inputs, padding=True, truncation=True, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    model.to(device)
    with torch.no_grad():
        logits = model(**tokenized_inputs).logits
    predicted_labels = [id2label[label.item()] for label in torch.argmax(logits, axis=1)]
    return predicted_labels


def load_prod_forecasting_model():
    model_checkpoint = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "final_finetuned_model_ForeSight")
    tokenizer_f = AutoTokenizer.from_pretrained(model_checkpoint)
    df = pd.read_json('./fine_tune_data_foresight.json')
    unique_labels_f = df['completion'].unique().tolist()
    id2label_f = {i: label for i, label in enumerate(unique_labels_f)}
    label2id_f = {label: i for i, label in enumerate(unique_labels_f)}
    model_f = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(unique_labels_f), id2label=id2label_f, label2id=label2id_f
    )
    return tokenizer_f, model_f, id2label_f


def get_prod_forecast(tokenizer_f, model_f, id2label_f, user_query, time_series_data):
    new_text_inputs = [f"{series} {user_query}" for series in time_series_data]
    tokenized_inputs = tokenizer_f(new_text_inputs, padding=True, truncation=True, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    model_f.to(device)
    with torch.no_grad():
        logits = model_f(**tokenized_inputs).logits

    # Post-process logits to ensure positive values
    predicted_indices = torch.argmax(logits, axis=1)
    predicted_labels = [id2label_f[label.item()] for label in predicted_indices]

    # Ensure predictions are numeric and positive
    positive_predictions = []
    for label in predicted_labels:
        try:
            numeric_label = float(label)
            positive_label = max(0, numeric_label)  # Replace negative values with 0
            positive_predictions.append(str(positive_label))
        except ValueError:
            positive_predictions.append(label)  # Keep as-is if conversion fails

    return positive_predictions


# **Function to execute LiNGAM and DiffAN causal discovery**
def run_causal_discovery():
    """Executes both LiNGAM and DiffAN causal discovery algorithms with user-selected dataset and features."""

    # **Debugging: Print stored session values before execution**
    st.write(f"📂 Uploaded Dataset Path: `{st.session_state['uploaded_file_path']}`")
    st.write(f"📝 Selected Features: `{st.session_state['selected_features']}`")

    if not st.session_state["uploaded_file_path"]:
        st.error("❌ No dataset uploaded. Please upload a dataset first.")
        return

    if not st.session_state["selected_features"] or st.session_state["selected_features"].strip() == "":
        st.error("⚠️ Please enter the features for causal analysis before performing discovery.")
        return

    ### **Step 1: Execute LiNGAM Causal Discovery**
    st.write("🟢 Running LiNGAM...")
    run_lingam()
    st.write("✅ LiNGAM completed.")

    ### **Step 2: Execute DiffAN Causal Discovery**
    st.write("🟡 Running DiffAN...")
    #run_diffan()  # Make sure this actually runs
    st.write("✅ DiffAN completed.")

# --- Function to answer causal queries ---
def answer_causal_query(question, causal_relations):
    question_lower = question

    if "does" in question_lower and "cause" in question_lower:
        match = re.match(r'does\s+(\w+)\s+cause\s+(\w+)\??', question_lower)
        if match:
            source, target = match.groups()
            if (source, target) in causal_relations:
                return f"✅ Yes, `{source}` causes `{target}`."
            else:
                return f"❌ No, `{source}` does not cause `{target}`."

    elif "is there a causal relation" in question_lower:
        match = re.findall(r'between\s+(\w+)\s+and\s+(\w+)', question_lower)
        if match:
            source, target = match[0]
            if (source, target) in causal_relations:
                return f"✅ Yes, `{source}` causes `{target}`."
            elif (target, source) in causal_relations:
                return f"✅ Yes, `{target}` causes `{source}`."
            else:
                return f"❌ No causal relation between `{source}` and `{target}` found."

    return "⚠️ I couldn't understand the causal query clearly. Please rephrase."

def run_lingam():
    import os
    import re
    import pickle
    import pandas as pd
    import networkx as nx
    from pyvis.network import Network

    # Utility to render graph
    def render_custom_graph(save_path="custom_lingam_graph.html", enable_download=True):
        # Use session-stored edges instead of file
        edge_list = st.session_state.get("lingam_edges", [])
        G = nx.DiGraph()

        # Add all edges
        for source, target in edge_list:
            G.add_edge(source, target)

        # ✅ Add all selected features as standalone nodes if not already in the graph
        selected_nodes = [n.strip() for n in st.session_state.get("selected_features", "").split(",") if n.strip()]
        tooltip_map = {}

        for node in selected_nodes:
            semantic_info = get_feature_semantic_info(node, rdf_graph)
            tooltip_map[node] = "<br>".join(semantic_info) if semantic_info else "No info from KG"
            G.add_node(node)  # Add node with no tooltip — just to preserve edge logic

        net = Network(height="800px", width="1400px", notebook=False, directed=True)
        net.from_nx(G)
        # Overwrite PyVis node tooltips using tooltip_map
        for node in net.nodes:
            node_id = node["id"]
            if node_id in tooltip_map:
                node["title"] = tooltip_map[node_id]

        net.set_options("""
            var options = {
              "edges": { "arrows": { "to": { "enabled": true } } },
              "nodes": { "shape": "dot", "size": 16 }
            }
        """)
        net.save_graph(save_path)

        # Display in Streamlit
        with open(save_path, "r", encoding="utf-8") as f:
            html_content_custom = f.read()
        st.components.v1.html(html_content_custom, height=800, width=1400, scrolling=True)

        if enable_download:
            with open(save_path, "rb") as f:
                st.download_button(
                    label="📥 Download Customized LiNGAM Graph",
                    data=f,
                    file_name=os.path.basename(save_path),
                    mime="text/html",
                    key=f"download_custom_lingam_{uuid.uuid4()}"
                )

    lingam_script_path = "lingam.py"
    modified_lingam_script = "modified_lingam.py"
    new_dataset_path = st.session_state["uploaded_file_path"]

    df = pd.read_csv(new_dataset_path)
    available_columns = df.columns.tolist()

    selected_features_list = [feat.strip() for feat in st.session_state["selected_features"].split(",")]
    valid_features = [feat for feat in selected_features_list if feat in available_columns]

    if not valid_features:
        st.error("❌ None of the selected features exist in the dataset.")
        return

    valid_features_str = ", ".join(f'"{feat}"' for feat in valid_features)

    # --- Modify and write modified_lingam.py ---
    with open(lingam_script_path, "r", encoding="utf-8") as file:
        script_content = file.read()

    script_content = re.sub(r'csv_file = ".*?"', f'csv_file = "{new_dataset_path}"', script_content)
    script_content = re.sub(
        r'df = pd.read_csv\(csv_file.*?\)',
        f'df = pd.read_csv(csv_file, usecols=[{valid_features_str}])',
        script_content,
        flags=re.DOTALL
    )

    if "feature_descriptions" in st.session_state:
        desc_entries = []
        for line in st.session_state["feature_descriptions"].splitlines():
            if ":" in line:
                key, rest = line.split(":", 1)
                if "," in rest:
                    category, description = rest.split(",", 1)
                    desc_entries.append(f'"{key.strip()}": ("{category.strip()}", "{description.strip()}")')
        node_desc_str = "NODE_DESCRIPTIONS = {\n    " + ",\n    ".join(desc_entries) + "\n}"
        if "NODE_DESCRIPTIONS" in script_content:
            script_content = re.sub(r'NODE_DESCRIPTIONS\s*=\s*{.*?}', node_desc_str, script_content, flags=re.DOTALL)
        else:
            lines = script_content.splitlines()
            insert_idx = max(i for i, line in enumerate(lines) if line.strip().startswith("import")) + 1
            lines.insert(insert_idx, "\n" + node_desc_str + "\n")
            script_content = "\n".join(lines)

    with open(modified_lingam_script, "w", encoding="utf-8") as file:
        file.write(script_content)

    os.system(f"python {modified_lingam_script}")

    lingam_edges_path = "lingam_graph_edges.pkl"
    if os.path.exists(lingam_edges_path):
        with open(lingam_edges_path, "rb") as f:
            original_edges = list(pickle.load(f))
    else:
        original_edges = []

    if "lingam_edges" not in st.session_state or not st.session_state["lingam_edges"]:
        st.session_state["lingam_edges"] = original_edges

    # Display LiNGAM HTML Graph (Optional)
    lingam_html_path = "lingam_causal_graph.html"
    if os.path.exists(lingam_html_path):
        with open(lingam_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.session_state["lingam_graph_html"] = html_content
        st.download_button("📥 Download LiNGAM Graph", data=html_content, file_name="lingam_causal_graph.html", mime="text/html",key=f"download_lingam_graph_{uuid.uuid4()}")
        st.components.v1.html(html_content, height=800, width=1400, scrolling=True)
    else:
        st.error("❌ LiNGAM graph not found.")

    # --- After LiNGAM execution, save causal relations ---
    st.session_state["causal_relations"] = [(source, target) for source, target in
                                            st.session_state.get("lingam_edges", [])]

    # --- Add NEW NODE section ---
    st.markdown("---")
    st.subheader("➕ Add a New Feature (Node) to the Causal Graph")

    with st.form("add_node_form_persistent"):
        new_node = st.text_input("Enter new feature name:")
        submitted_node = st.form_submit_button("Add Feature and Re-run LiNGAM")

    if submitted_node:
        if new_node:
            if new_node not in df.columns:
                st.error(f"❌ `{new_node}` does not exist in dataset columns.")
            else:
                current_features = [f.strip() for f in st.session_state["selected_features"].split(",")]
                if new_node not in current_features:
                    current_features.append(new_node)
                    st.session_state["selected_features"] = ", ".join(current_features)
                    st.success(f"✅ Feature `{new_node}` added! Re-running LiNGAM...")
                    st.rerun()
                else:
                    st.info("ℹ️ Feature already included.")
                    render_custom_graph()  # render current graph

    # --- Add edge section ---
    st.subheader("✏️ Modify LiNGAM Causal Graph")
    with st.form("add_edge_form_persistent"):
        col1, col2 = st.columns(2)
        with col1:
            source_node = st.text_input("Source Node (cause)")
        with col2:
            target_node = st.text_input("Target Node (effect)")
        submitted = st.form_submit_button("➕ Add Edge")

    if submitted:
        if source_node and target_node:
            new_edge = (source_node.strip(), target_node.strip())
            if new_edge not in st.session_state["lingam_edges"]:
                st.session_state["lingam_edges"].append(new_edge)

                # 🔽 SAVE updated edges to disk
                with open("lingam_graph_edges.pkl", "wb") as f:
                    pickle.dump(st.session_state["lingam_edges"], f)

                st.success(f"✅ Edge added: {source_node.strip()} ➝ {target_node.strip()}")
            else:
                st.info("ℹ️ Edge already exists.")

            render_custom_graph()  # always render updated graph
        else:
            st.error("⚠️ Please enter both source and target nodes.")

    # Show graph if any edges exist
    if st.session_state["lingam_edges"]:
        st.subheader("📈 Updated Causal Graph with Custom Edges")
        render_custom_graph()


# **Function to execute DiffAN causal discovery**
def run_diffan():
    """Executes DiffAN causal discovery algorithm with user-selected dataset and features."""

    st.write("🚀 Starting DiffAN causal discovery...")

    # Define paths
    # diffan_script_path = "/Users/chathurangishyalika/Custom_Compact_Copilot/SmartPilot/Agent 3: InfoGuide/src/DiffAN/swat_viz.py"
    # diffan_output_dir = "/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Agent 3: InfoGuide/src/DiffAN"
    # new_dataset_path = "/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Agent 3: InfoGuide/src/uploaded_dataset.csv"  # ✅ Absolute Path
    # modified_diffan_script = os.path.join(diffan_output_dir, "modified_diffan.py")
    # Get the current directory where the script is located

    # Define paths relative to the script location
    diffan_script_path = os.path.join(script_dir, "DiffAN", "swat_viz.py")
    diffan_output_dir = os.path.join(script_dir, "DiffAN")
    new_dataset_path = os.path.join(script_dir, "uploaded_dataset.csv")
    modified_diffan_script = os.path.join(diffan_output_dir, "modified_diffan.py")

    # ✅ Expected output files
    causal_graph_residue = os.path.join(diffan_output_dir, "causal_graph_residue.html")
    causal_graph_no_residue = os.path.join(diffan_output_dir, "causal_graph_no_residue.html")

    # ✅ Check if the original DiffAN script exists
    if not os.path.exists(diffan_script_path):
        st.error(f"❌ Error: Original DiffAN script `{diffan_script_path}` not found!")
        return

    # ✅ Read the existing script
    try:
        with open(diffan_script_path, "r", encoding="utf-8") as file:
            script_content = file.read()
    except Exception as e:
        st.error(f"❌ Error reading DiffAN script: {e}")
        return

    # ✅ Correctly replace dataset path in script
    script_content = re.sub(r'csv_file = ".*?"', f'csv_file = "{new_dataset_path}"', script_content)

    # ✅ Replace selected features
    selected_features_list = [f'"{feat.strip()}"' for feat in st.session_state["selected_features"].split(",")]
    selected_features_str = ", ".join(selected_features_list)
    script_content = re.sub(
        r'selected_columns = \[.*?\]',
        f'selected_columns = [{selected_features_str}]',
        script_content,
        flags=re.DOTALL
    )

    # ✅ Format and inject NODE_DESCRIPTIONS with category and description
    if "feature_descriptions" in st.session_state:
        descriptions_lines = st.session_state["feature_descriptions"].splitlines()
        desc_dict_entries = []

        for line in descriptions_lines:
            if ":" in line:
                key, rest = line.split(":", 1)
                key = key.strip()
                rest = rest.strip()

                # Expecting: Category, Description
                if "," in rest:
                    category, description = rest.split(",", 1)
                    category = category.strip()
                    description = description.strip()
                    desc_dict_entries.append(f'"{key}": ("{category}", "{description}")')

        node_desc_str = "NODE_DESCRIPTIONS = {\n    " + ",\n    ".join(desc_dict_entries) + "\n}"

        # Replace existing NODE_DESCRIPTIONS or inject if missing
        if "NODE_DESCRIPTIONS" in script_content:
            script_content = re.sub(
                r'NODE_DESCRIPTIONS\s*=\s*{.*?}',
                node_desc_str,
                script_content,
                flags=re.DOTALL
            )
        else:
            # Inject near the top of the script
            script_content = node_desc_str + "\n\n" + script_content

    # ✅ Write the modified script
    try:
        with open(modified_diffan_script, "w", encoding="utf-8") as file:
            file.write(script_content)
        st.write(f"✅ Modified DiffAN script created successfully at `{modified_diffan_script}`")
    except Exception as e:
        st.error(f"❌ Error writing modified DiffAN script: {e}")
        return

    # ✅ Ensure script is properly created
    if not os.path.exists(modified_diffan_script):
        st.error("❌ Error: Modified DiffAN script was not created!")
        return

    # ✅ Run DiffAN script
    try:
        result = subprocess.run(
            ["python", modified_diffan_script],
            cwd=diffan_output_dir,
            capture_output=True, text=True, check=True
        )
        st.write("✅ DiffAN script execution completed.")
        st.write("📄 DiffAN Output Log:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error Running DiffAN Script:\n{e.stderr}")
        return

    # ✅ Debug: List generated files
    st.write("🔍 Checking for generated files in DiffAN directory:")
    generated_files = glob.glob(os.path.join(diffan_output_dir, "*"))
    st.write(generated_files)

    # ✅ **Display & Download Causal Graphs**
    found_graphs = False

    # 🔵 **Check for `causal_graph_residue.html`**
    # For causal_graph_residue.html:
    if os.path.exists(causal_graph_residue):
        found_graphs = True
        st.success("✅ DiffAN Causal Discovery (With Residue) Completed!")
        with open(causal_graph_residue, "rb") as file:
            st.download_button(
                label="📥 Download Causal Graph (With Residue)",
                data=file,
                file_name="causal_graph_residue.html",
                mime="text/html",
            )
        with open(causal_graph_residue, "r", encoding="utf-8") as f:
            html_content_residue = f.read()
        st.session_state["diffan_graph_residue_html"] = html_content_residue
        st.components.v1.html(st.session_state["diffan_graph_residue_html"], height=800, width=1400, scrolling=True)

    # 🟢 **Check for `causal_graph_no_residue.html`**
    # For causal_graph_no_residue.html:
    if os.path.exists(causal_graph_no_residue):
        found_graphs = True
        st.success("✅ DiffAN Causal Discovery (Without Residue) Completed!")
        with open(causal_graph_no_residue, "rb") as file:
            st.download_button(
                label="📥 Download Causal Graph (Without Residue)",
                data=file,
                file_name="causal_graph_no_residue.html",
                mime="text/html",
            )
        with open(causal_graph_no_residue, "r", encoding="utf-8") as f:
            html_content_no_residue = f.read()
        st.session_state["diffan_graph_no_residue_html"] = html_content_no_residue
        st.components.v1.html(st.session_state["diffan_graph_no_residue_html"], height=800, width=1400, scrolling=True)

    if not found_graphs:
        st.error("❌ Error: DiffAN causal discovery visualization files were not found.")


# UI Title
st.title("SmartPilot: Agent-Based CoPilot for Intelligent Manufacturing")


# Sidebar with Sample Questions
st.sidebar.title("📌 Sample Questions")

with st.sidebar:
    if st.button("Upload your dataset", key="causal_agent_queries"):
        st.session_state["selected_question"] = "Upload your dataset"

    st.subheader("📖 Documentation Queries")
    documentation_queries = [
        "How to set up the toy rocket manufacturing machine?",
        "What are the safety protocols for the manufacturing process?",
        "How to troubleshoot common issues in the manufacturing pipeline?",
        "Describe the maintenance procedure for the assembly line machines.",
        "What are the steps to calibrate the sensors in the manufacturing setup?",
        "How to perform a quality check on the manufactured toy rockets?",
        "What materials are needed for the manufacturing process?",
        "How to store and handle materials safely?",
        "What are the emergency procedures in case of a malfunction?",
        "How to document the production cycle for future reference?"
    ]
    for query in documentation_queries:
        if st.button(query, key=query):
            st.session_state["selected_question"] = query

    st.subheader("📊 Anomaly Prediction Queries")
    if st.button("Enter the current sensor values to check if an anomaly will happen next", key="anomaly_query"):
        st.session_state[
            "selected_question"] = "Enter the current sensor values to check if an anomaly will happen next"

    st.subheader("📈 Production Forecasting Queries")
    if st.button("Enter the current production statistics to get the next production", key="prod_forecast_query"):
        st.session_state["selected_question"] = "Enter the current production statistics to get the next production"

    st.subheader("🧩 Causal Analysis Queries")
    if st.button("Select the features for causal analysis", key="causal_analysis_features"):
        st.session_state["selected_question"] = "Select the features for causal analysis"
    if st.button("Give Feature Descriptions", key="feature_description"):
        st.session_state["selected_question"] = "Give Feature Descriptions"
    if st.button("Perform Causal Discovery", key="causal_discovery"):
        st.session_state["selected_question"] = "Perform Causal Discovery"
    if st.button("🧠 Run Root Cause Analysis"):
        if "uploaded_data" not in st.session_state or st.session_state["uploaded_data"] is None:
            st.warning("Upload dataset first.")
        elif adj_matrix is None or node_labels is None:
            st.warning("Run causal discovery first.")
        else:
            df = st.session_state["uploaded_data"]
            # sensor_ranges = parse_sensor_ranges("/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Agent 3: InfoGuide/src/assets/sensor_cycle_ranges.txt")
        
            sensor_path = os.path.join(script_dir, "assets", "sensor_cycle_ranges.txt")
            sensor_ranges = parse_sensor_ranges(sensor_path)
            rca_results = analyze_all(df, sensor_ranges, total_effects, node_labels)

            st.session_state["rca_results"] = rca_results

            st.write("🔍 RCA Results Raw:", rca_results)

            st.subheader("🔍 Root Cause Analysis Results")
            for i, result in enumerate(rca_results):
                if result["anomalous_sensors"]:
                    st.markdown(
                        f"### Row {i + 1}: ❗ State = `{result['actual_state']}` at Cycle {result['CycleState']}")
                    st.markdown(f"**Anomalous Sensors:** {', '.join(result['anomalous_sensors'])}")
                    for sensor, causes in result["root_cause_paths"].items():
                        if causes:
                            for parent, strength in causes:
                                st.markdown(f"📌 `{parent}` ➝ `{sensor}` (strength: {strength:.4f})")
                    st.markdown("---")




# **Display Dataset Statistics in Main Window**
if st.session_state["uploaded_data"] is not None:
    df = st.session_state["uploaded_data"]

    st.subheader("📊 Dataset Overview")

    # Displaying basic dataset details
    st.write(f"**🟢 Number of Rows:** {df.shape[0]}")
    st.write(f"**🟢 Column Names:** {', '.join(df.columns)}")

    # Display descriptive statistics using df.describe()
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe())

# **Upload Dataset Feature + Sample Dataset Buttons**
if st.session_state["selected_question"] == "Upload your dataset":
    st.subheader("📂 Upload Your Dataset")

    # Upload user dataset
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        file_path = f"uploaded_dataset.{file_extension}"

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load to session
        st.session_state["uploaded_data"] = pd.read_csv(file_path) if file_extension == "csv" else pd.read_excel(file_path)
        st.session_state["uploaded_file_path"] = file_path
        st.session_state["selected_question"] = None
        st.success(f"✅ Dataset uploaded successfully! Stored as `{file_path}`")

    st.markdown("---")

    # Sample Dataset Buttons
    st.subheader("📊 Or Try a Sample Dataset")
    col1, col2 = st.columns(2)
    # dest_path = "/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Agent 3: InfoGuide/src/uploaded_dataset.csv"
    dest_path = os.path.join(os.path.dirname(__file__), "uploaded_dataset.csv")

    with col1:
        if st.button("📎 Secure Water Treatment (SWaT) Dataset"):
            # sample_path = "/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Sample datasets/swat_sample.csv"
            sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "Sample datasets", "swat_sample.csv")

            try:
                shutil.copy(sample_path, dest_path)
                st.session_state["uploaded_data"] = pd.read_csv(sample_path)
                st.session_state["uploaded_file_path"] = sample_path
                st.session_state["selected_question"] = None
                st.success("✅ SWAT sample dataset loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load SWAT dataset: {e}")

    with col2:
        if st.button("📎 Future Factories (FF) Analog Dataset "):
            # sample_path = "/Users/ledarssan/AIISC/CCN_models/SMP-Deploy/SmartPilot_AIISC/Sample datasets/FF_test_sample.csv"
            sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "Sample datasets", "FF_test_sample.csv")
            try:
                shutil.copy(sample_path, dest_path)
                st.session_state["uploaded_data"] = pd.read_csv(sample_path)
                st.session_state["uploaded_file_path"] = sample_path
                st.session_state["selected_question"] = None
                st.success("✅ FF sample dataset loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load FF dataset: {e}")


# **Feature Selection for Causal Analysis**
# **Feature Selection for Causal Analysis**
if st.session_state["selected_question"] == "Select the features for causal analysis":
    st.subheader("📝 Select Features for Causal Analysis")

    # **Ensure the session state is properly initialized**
    if "selected_features_temp" not in st.session_state:
        st.session_state["selected_features_temp"] = st.session_state["selected_features"]

    # **Use a text input box that updates session state instantly**
    def update_selected_features():
        st.session_state["selected_features"] = st.session_state["selected_features_temp"]

    selected_features = st.text_area(
        "Enter the feature names separated by commas (e.g., FIT101, MV101, P101):",
        value=st.session_state["selected_features_temp"],
        key="selected_features_temp",
        on_change=update_selected_features  # Updates `selected_features` when modified
    )

    if st.button("Save Selected Features"):
        if st.session_state["selected_features_temp"].strip():
            st.session_state["selected_features"] = st.session_state["selected_features_temp"].strip()
            st.success(f"✅ Selected features saved: {st.session_state['selected_features']}")
        else:
            st.error("❌ Please enter at least one feature.")

    # Reset `selected_question` after saving
    if st.session_state["selected_features"]:
        st.session_state["selected_question"] = None


# **Show the stored dataset path and selected features**
if st.session_state["uploaded_file_path"]:
    st.write(f"📂 **Dataset Path:** `{st.session_state['uploaded_file_path']}`")

if st.session_state["selected_features"]:
    st.write(f"📝 **Selected Features:** `{st.session_state['selected_features']}`")

# Feature Description Input Section
if st.session_state["selected_question"] == "Give Feature Descriptions":
    st.subheader("🧾 Describe Features for Causal Analysis")

    # Initialize description temp storage
    if "feature_descriptions_temp" not in st.session_state:
        st.session_state["feature_descriptions_temp"] = st.session_state.get("feature_descriptions", "")

    def update_feature_descriptions():
        st.session_state["feature_descriptions"] = st.session_state["feature_descriptions_temp"]

    feature_descriptions = st.text_area(
        "Enter feature descriptions (e.g., FIT101: Flow sensor at inlet pipe):",
        value=st.session_state["feature_descriptions_temp"],
        key="feature_descriptions_temp",
        on_change=update_feature_descriptions
    )

    if st.button("Save Feature Description"):
        if st.session_state["feature_descriptions_temp"].strip():
            st.session_state["feature_descriptions"] = st.session_state["feature_descriptions_temp"].strip()
            st.success("✅ Feature descriptions saved.")
        else:
            st.error("❌ Please enter descriptions before saving.")

# **Perform Causal Discovery**
if st.session_state["selected_question"] == "Perform Causal Discovery":
    if st.session_state["uploaded_file_path"] and st.session_state["selected_features"] and st.session_state["selected_features"].strip():
        st.subheader("🔍 Running Causal Discovery...")
        run_causal_discovery()  # Execute the modified LiNGAM causal discovery script
    else:
        st.error("⚠️ Please upload a dataset and select features before performing causal discovery.")

    #st.session_state["selected_question"] = None  # Reset selection




# Display selected question if available
# Add a floating header for the "Selected Question"
st.markdown(
    """
    <style>
    .selected-question-container {
        position: fixed;
        top: 60px; /* Adjust this if needed */
        left: 50%;
        transform: translateX(-50%);
        background-color: white;
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the selected question in a fixed header
if st.session_state["selected_question"]:
    st.markdown(f"<div class='selected-question-container'>🔹 Selected Question: {st.session_state['selected_question']}</div>", unsafe_allow_html=True)



# Chat Input
user_input = st.chat_input("Enter your question...")

# If no user input, but a question was selected from sidebar, use the selected question
if not user_input and st.session_state["selected_question"]:
    if st.session_state["selected_question"] not in ["Upload your dataset", "Give Feature Descriptions","Select the features for causal analysis","Perform Causal Discovery"]:
        user_input = st.session_state["selected_question"]
        st.session_state["selected_question"] = None  # Clear after use

# Load models
tokenizer, model, id2label = load_anomaly_prediction_model()
tokenizer_f, model_f, id2label_f = load_prod_forecasting_model()

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    user_query_lower = user_input.lower()
    response = None

    # 🟦 Robot sensor query
    robot_number, confidence = detect_robot_sensor_query(user_input)
    if robot_number:
        response = get_sensors_connected_to_robot(robot_number, rdf_graph,
                                                  st.session_state.get("robot_label_to_uuid", {}))

    # 🟨 Anomaly type query
    elif detect_anomaly_type_query_semantic(user_input):
        anomaly_types = get_anomaly_types_from_kg(rdf_graph)
        if anomaly_types:
            response = "🧠 Types of anomalies in the Knowledge Graph:\n\n" + "\n- ".join([""] + anomaly_types)
        else:
            response = "⚠️ No anomaly types found in the Knowledge Graph."

    # 🟥 Causal reasoning
    elif (causal_response := handle_causal_reasoning_query(user_input, adj_matrix, node_labels)):
        response = causal_response

    # 🟠 Root Cause Analysis Queries (semantic match)
    elif (
            "root cause" in user_input.lower()
            and "rca_results" in st.session_state
            and st.session_state["rca_results"]
    ):
        response = answer_root_cause_query(user_input, st.session_state["rca_results"])

    # 🟩 Causal yes/no relation
    elif any(kw in user_query_lower for kw in
             ["does", "cause", "causal relation"]) and "causal_relations" in st.session_state:
        response = answer_causal_query(user_input, st.session_state["causal_relations"])

    # 🔵 Anomaly prediction
    elif "anomaly" in user_query_lower and re.search(r'\d', user_query_lower):
        predicted_labels = get_anomaly_prediction(tokenizer, model, id2label, user_input, ["[0. 0. 0.]"])
        response = f"Predicted anomaly labels: {', '.join(predicted_labels)}"

    # 🟣 Production forecasting
    elif "production" in user_query_lower and re.search(r'\d', user_query_lower):
        predicted_labels = get_prod_forecast(tokenizer_f, model_f, id2label_f, user_input, ["[0. 0. 0.]"])
        response = f"Predicted product values: {', '.join(predicted_labels)}"

    elif st.session_state.get("ProcessOntologyQa") is not None:
        process_qa = st.session_state["ProcessOntologyQa"]
        lowered = user_input.lower().strip()
        process_response = None

        # Robot function
        if "function of robot" in lowered:
            match = re.search(r"function of (?:the )?robot (\d+)", lowered)
            if match:
                process_response = process_qa.get_robot_function(f"Robot {match.group(1)}")

        # Gripper of robot
        elif "gripper of robot" in lowered:
            match = re.search(r"gripper of (?:the )?robot (\d+)", lowered)
            if match:
                process_response = process_qa.get_gripper_of_robot(f"Robot {match.group(1)}")

        # Sensors of robot
        elif "sensors attached to robot" in lowered:
            match = re.search(r"sensors attached to (?:the )?robot (\d+)", lowered)
            if match:
                s = process_qa.get_sensors_of_robot(f"Robot {match.group(1)}")
                process_response = ["📡 Sensors connected:"] + s

        # Sensor description / function
        elif re.search(r"(function|description) of (?:the )?sensor", lowered):
            match = re.search(r"(?:function|description) of (?:the )?sensor ([\w\d_ ]+)", lowered)
            if match:
                sensor_name = match.group(1).strip()
                sensor_desc = process_qa.get_sensor_description(sensor_name)
                if sensor_desc:
                    process_response = [sensor_desc]


        # Sensor value range
        elif "range of" in lowered:
            match = re.search(r"range of (?:the )?([\w\d_]+)", lowered)
            if match:
                process_response = [process_qa.get_sensor_value_range(match.group(1).strip())]

        # Trace path from robot
        elif "trace" in lowered and "robot" in lowered:
            match = re.search(r"robot (\d+)", lowered)
            if match:
                paths = process_qa.trace_robot_to_sensor_value(f"Robot {match.group(1)}")
                process_response = ["🔍 Trace Paths:"] + paths

        # Anomalies associated with sensor
        elif "anomalies associated with sensor" in lowered:
            match = re.search(r"sensor ([\w\d_]+)", lowered)
            if match:
                process_response = process_qa.get_anomalies_for_sensor(match.group(1))

        # Sensor types or sensor list
        elif "types of sensors" in lowered or "details on the sensors" in lowered:
            process_response = process_qa.get_all_sensor_details()

        # Fallback
        if process_response:
            if isinstance(process_response, str):
                process_response = [process_response]
            if len(process_response) == 1:
                response = f"📘 Process Ontology Answer:\n{process_response[0]}"
            else:
                response = "📘 Process Ontology Answer:\n- " + "\n- ".join(process_response)

        # ⚫ LLM fallback + feature info
        # ✅ Fallback from ProcessOntologyQA to LLM if nothing matched
        if process_response is None:
            data = Knowledge_Representation.organize_data(AssetLoader.read_data())
            context = Retr.retrieve_context(data, user_input, symb_model=Symbolic_Model(), top_k=1)[0]

            # Enrich with KG semantic info
            if "selected_features" in st.session_state:
                kg_descriptions = []
                for feature in st.session_state["selected_features"].split(","):
                    desc = get_full_feature_semantic_info(feature.strip(), rdf_graph)
                    if desc:
                        kg_descriptions.append(f"{feature.strip()}:\n" + "\n".join(desc))
                if kg_descriptions:
                    context += "\n\n---\n📘 Feature Semantic Info from KG:\n" + "\n\n".join(kg_descriptions)

            # Enrich with Process Ontology info
            # Add Process Ontology entity descriptions
            if "ProcessOntologyQa" in st.session_state:
                process_qa = st.session_state["ProcessOntologyQa"]
                ontology_data = process_qa.get_ontology_data()
                process_ontology_info = []
                for feature in st.session_state.get("selected_features", "").split(","):
                    feature_clean = feature.strip()
                    if feature_clean:
                        info = get_full_entity_semantic_info(feature_clean, ontology_data)
                        if info:
                            process_ontology_info.append(f"{feature_clean}:\n" + "\n".join(info))
                if process_ontology_info:
                    context += "\n\n---\n🏭 Feature Info from Process Ontology:\n" + "\n\n".join(process_ontology_info)

            # Extra context: auto-include all Sensor_Value nodes
            sensor_value_info = []
            for node in ontology_data.get("nodes", []):
                if node.get("type") == "Sensor_Value":
                    val_name = node.get("item_name", "Unknown")
                    min_val = node.get("min_value", "N/A")
                    max_val = node.get("max_value", "N/A")
                    unit = node.get("unit", "N/A")
                    sensor_value_info.append(f"{val_name}:\n📏 Tolerance Range: {min_val} to {max_val} (Unit: {unit})")
            sensor_value_info.sort()

            if sensor_value_info:
                context += "\n\n---\n📈 Sensor Tolerance Limits:\n" + "\n\n".join(sensor_value_info)

            # Run LLM
            llm = LLM()

            # Mentioned entity in query (like "who manufactures Robot 4")
            ontology_nodes = ontology_data.get("nodes", [])
            mentioned_entity = extract_entity_name(user_input, ontology_nodes)

            if mentioned_entity:
                mentioned_info = get_full_entity_semantic_info(mentioned_entity, ontology_data)
                if mentioned_info:
                    context += f"\n\n---\n🏭 Info on `{mentioned_entity}` from Process Ontology:\n" + "\n".join(
                        mentioned_info)

            llm.set_prompt(AssetLoader.get_templates().get("documentation_agent", ""), user_input, context)
            response = llm.respond_to_prompt()

        else:
            if isinstance(process_response, str):
                process_response = [process_response]
            if len(process_response) == 1:
                response = f"📘 Process Ontology Answer:\n{process_response[0]}"
            else:
                response = "📘 Process Ontology Answer:\n- " + "\n- ".join(process_response)

    # Always append the response if available
    if response:
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # 💡 Handle "What if I add the variable..." logic
        match = re.search(r'what.*add.*variable\s+"?([\w\d_]+)"?', user_input)
        if match:
            feature_to_add = match.group(1).strip()

            if st.session_state["uploaded_data"] is not None:
                df = st.session_state["uploaded_data"]
                available_columns = df.columns.tolist()

                if feature_to_add in available_columns:
                    current_features = [f.strip() for f in st.session_state.get("selected_features", "").split(",") if f.strip()]

                    if feature_to_add not in current_features:
                        current_features.append(feature_to_add)
                        st.session_state["selected_features"] = ", ".join(current_features)

                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"🔍 Adding `{feature_to_add}` to causal discovery and updating LiNGAM graph..."
                        })

                        run_lingam()

                    else:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"ℹ️ `{feature_to_add}` is already included in the causal analysis."
                        })
                else:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"❌ Feature `{feature_to_add}` does not exist in the uploaded dataset."
                    })
            else:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "❌ No dataset uploaded. Please upload a dataset first."
                })


# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
