from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Path to the directory containing adapter_config.json and adapter_model.safetensors
adapter_path = "content/final_finetuned_model"

# Load base model and tokenizer (must match the base model used for fine-tuning)
base_model_id = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Attempt to load the adapter
try:
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✅ Adapter loaded successfully!")

    # Run a basic inference to verify
    input_text = "Translate English to French: Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=20)
    print("🧪 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print("❌ Failed to load adapter.")
    print(e)
