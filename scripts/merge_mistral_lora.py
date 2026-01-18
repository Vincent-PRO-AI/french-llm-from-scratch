import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def merge_and_save(base_model_id, lora_path, output_path):
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model in FP16 for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    
    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--lora", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    merge_and_save(args.base, args.lora, args.out)
