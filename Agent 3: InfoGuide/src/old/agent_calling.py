import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM
import pandas as pd
import os


# Fix: Initialize session state keys before usage
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = " "

if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None


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
    predicted_labels = [id2label_f[label.item()] for label in torch.argmax(logits, axis=1)]
    return predicted_labels


# UI Title
st.title("SmartPilot: Agent-Based CoPilot for Intelligent Manufacturing")

# Sidebar with Sample Questions
st.sidebar.title("📌 Sample Questions")

with st.sidebar:
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
        st.session_state["selected_question"] = "Enter the current sensor values to check if an anomaly will happen next"

    st.subheader("📈 Production Forecasting Queries")
    if st.button("Enter the current production statistics to get the next production", key="prod_forecast_query"):
        st.session_state["selected_question"] = "Enter the current production statistics to get the next production"


# Display selected question if available
if st.session_state["selected_question"]:
    st.write(f"🔹 **Selected Question:** {st.session_state['selected_question']}")

# Chat Input
user_input = st.chat_input("Enter your question...")

# If no user input, but a question was selected from sidebar, use the selected question
if not user_input and st.session_state["selected_question"]:
    user_input = st.session_state["selected_question"]
    st.session_state["selected_question"] = None  # Clear after use

# Load models
tokenizer, model, id2label = load_anomaly_prediction_model()
tokenizer_f, model_f, id2label_f = load_prod_forecasting_model()


# Function to check if text contains numbers
def contains_numbers(text):
    return bool(re.search(r'\d', text))


if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    user_query_lower = user_input.lower()

    try:
        if "anomaly" in user_query_lower and any(word in user_query_lower for word in ["what", "status"]) and contains_numbers(user_query_lower):
            input_parts = user_input.split(";")
            time_series_data = input_parts[0].strip().split(",") if len(input_parts) > 1 else ["[0. 0. 0.]"]
            user_query_text = input_parts[1].strip() if len(input_parts) > 1 else user_input
            predicted_labels = get_anomaly_prediction(tokenizer, model, id2label, user_query_text, time_series_data)
            response = f"Predicted anomaly labels: {', '.join(predicted_labels)}"

        elif "production" in user_query_lower and "what" in user_query_lower and contains_numbers(user_query_lower):
            input_parts = user_input.split(";")
            time_series_data = input_parts[0].strip().split(",") if len(input_parts) > 1 else ["[0. 0. 0.]"]
            user_query_text = input_parts[1].strip() if len(input_parts) > 1 else user_input
            predicted_labels = get_prod_forecast(tokenizer_f, model_f, id2label_f, user_query_text, time_series_data)
            response = f"Predicted product values: {', '.join(predicted_labels)}"

        else:
            data = Knowledge_Representation.organize_data(AssetLoader.read_data())
            context = st.session_state.conversation_history + \
                      Retr.retrieve_context(data, user_input, symb_model=Symbolic_Model(), top_k=1)[0]

            system_template = AssetLoader.get_templates().get("documentation_agent", "")
            llm = LLM()
            llm.set_prompt(system_template, user_input, context)
            response = llm.respond_to_prompt()

    except Exception as e:
        response = f"Error in processing input: {str(e)}"

    st.session_state["messages"].append({"role": "assistant", "content": response})

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div style='white-space: pre-line;'>{msg['content']}</div>", unsafe_allow_html=True)


