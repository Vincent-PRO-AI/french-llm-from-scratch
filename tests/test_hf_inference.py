#!/usr/bin/env python3
"""
Test rapide de téléchargement et génération depuis Hugging Face Hub
Modèle: vincent-pro-ai/french-llm-from-scratch
"""
import os
import sys

def main():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        print("❌ Transformers non installé. Installez avec: pip install transformers safetensors")
        print(f"Erreur: {e}")
        sys.exit(1)

    repo_id = os.environ.get("HF_REPO_ID", "vincent-pro-ai/french-llm-from-scratch")
    print(f"📥 Téléchargement depuis Hugging Face Hub: {repo_id}")

    # Chargement
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()

    # Prompt de test
    prompt = os.environ.get("HF_TEST_PROMPT", "Explique brièvement la différence entre apprentissage supervisé et non supervisé en français.")
    print(f"\n📝 Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", 96)),
            temperature=float(os.environ.get("HF_TEMPERATURE", 0.8)),
            top_p=float(os.environ.get("HF_TOP_P", 0.9)),
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n✅ Résultat:\n")
    print(result)

    # Sanity check minimal
    assert len(result) > len(prompt), "Le texte généré semble vide ou identique au prompt."
    print("\n🎉 Test OK")


if __name__ == "__main__":
    main()
