from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

adapter_path = "content/final_finetuned_model"

# Load adapter config to get correct base model
config = PeftConfig.from_pretrained(adapter_path)
print("Base model required:", config.base_model_name_or_path)

# Now use the correct base model
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Test inference
inputs = tokenizer("translate English to German: The weather is nice today.", return_tensors="pt")
output = model.generate(**inputs)
print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
