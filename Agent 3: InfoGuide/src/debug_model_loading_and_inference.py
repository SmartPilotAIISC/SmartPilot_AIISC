import time
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)

def main():
    start = time.time()
    logging.info("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    logging.info(f"Base model loaded in {time.time() - start:.2f}s")

    start = time.time()
    logging.info("Loading adapter...")
    model = PeftModelForSeq2SeqLM.from_pretrained(base_model, "content/final_finetuned_model")
    logging.info(f"Adapter loaded in {time.time() - start:.2f}s")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    logging.info("Running inference...")
    input_text = "translate English to German: The weather is nice today."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    logging.info(f"Output: {output}")

if __name__ == "__main__":
    main()
