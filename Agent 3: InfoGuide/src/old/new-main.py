import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM
import pandas as pd
import os

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

def prod_forecasting_model():
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


# Streamlit UI
st.title("SmartPilot:Agent-Based CoPilot for Intelligent Manufacturing")

# Sidebar for user role selection
st.sidebar.title("🛠 User Simulation")
users_and_queries = AssetLoader.get_queries()
user_roles = list(users_and_queries.keys())
selected_role = st.sidebar.selectbox("Select User Role", user_roles)
user_query = st.sidebar.selectbox("Select Query", users_and_queries[selected_role])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.write("**Chat with the AI:**")
user_input = st.chat_input("Enter time-series data (comma-separated) and ask your question...")

tokenizer, model, id2label = load_anomaly_prediction_model()
tokenizer_f, model_f, id2label_f = prod_forecasting_model()

if user_input or st.sidebar.button("Run Simulation"):
    st.session_state["messages"].append({"role": "user", "content": user_input or user_query})

    if selected_role == 'Anomaly Prediction and Sensor Values':
        try:
            input_parts = user_input.split(";")
            time_series_data = input_parts[0].strip().split(",") if len(input_parts) > 1 else ["[0. 0. 0.]"]
            user_query_text = input_parts[1].strip() if len(input_parts) > 1 else user_query

            predicted_labels = get_anomaly_prediction(tokenizer, model, id2label, user_query_text, time_series_data)
            response = f"Predicted anomaly labels: {', '.join(predicted_labels)}"
        except Exception as e:
            response = f"Error in processing input: {str(e)}"
    elif selected_role == 'Production Forecasting':
        try:
            input_parts = user_input.split(";")
            time_series_data = input_parts[0].strip().split(",") if len(input_parts) > 1 else ["[0. 0. 0.]"]
            user_query_text = input_parts[1].strip() if len(input_parts) > 1 else user_query

            predicted_labels = get_prod_forecast(tokenizer_f, model_f, id2label_f, user_query_text, time_series_data)
            response = f"Predicted product values: {', '.join(predicted_labels)}"
        except Exception as e:
            response = f"Error in processing input: {str(e)}"
    else:
        data = Knowledge_Representation.organize_data(AssetLoader.read_data())
        context = Retr.retrieve_context(data, user_query, symb_model=Symbolic_Model(), top_k=1)[0]
        system_template = AssetLoader.get_templates()[selected_role]
        llm = LLM()
        llm.set_prompt(system_template, user_query, context)
        response = llm.respond_to_prompt()

    st.session_state["messages"].append({"role": "assistant", "content": response})

# Inject custom CSS for chat UI enhancements
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        max-height: 500px;
    }
    .user-bubble {
        background-color: #73000a;
        color: white;
        border: 1px solid #73000a;
        border-radius: 15px;
        padding: 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #f0f0f0;
        color: black;
        border-radius: 15px;
        padding: 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        float: left;
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)

# Display chat messages with styled bubbles
# Display chat messages using st.chat_message() for user and assistant icons
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div style='white-space: pre-line;'>{msg['content']}</div>", unsafe_allow_html=True)


