#!/usr/bin/env python3
"""
🔄 PHASE 4 - Export GGUF: Conversion pour LM Studio
Exporte le modèle Mistral au format GGUF pour utilisation dans LM Studio.

Steps:
  1. Sauvegarder modèle au format HuggingFace standard
  2. Cloner llama.cpp si nécessaire
  3. Convertir en GGUF avec quantization (Q4_K_M recommandé)
  4. Tester le modèle dans LM Studio
"""

import os
import json
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Vérifier les dépendances requises."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ Transformers installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def ensure_model_is_hf_format(model_path: str):
    """S'assurer que le modèle est au format HuggingFace standard."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"📥 Vérifier modèle HuggingFace: {model_path}")
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.error(f"❌ Model directory not found: {model_path}")
            return False
        
        # Vérifier fichiers essentiels
        required_files = [
            "config.json",
            "model.safetensors" if (model_dir / "model.safetensors").exists() else "pytorch_model.bin",
            "tokenizer.json" if (model_dir / "tokenizer.json").exists() else "tokenizer.model"
        ]
        
        for file in required_files:
            file_path = model_dir / file
            if file_path.exists():
                logger.info(f"  ✅ Found: {file} ({file_path.stat().st_size / 1e6:.1f} MB)")
        
        # Charger pour vérifier
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"✅ Modèle chargeable avec transformers")
            del model, tokenizer
            return True
        except Exception as e:
            logger.error(f"❌ Modèle non chargeable: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur vérification modèle: {e}")
        return False

def setup_llama_cpp(llama_cpp_path: str = "./llama.cpp"):
    """Cloner et compiler llama.cpp si nécessaire."""
    try:
        llama_cpp_dir = Path(llama_cpp_path)
        
        if llama_cpp_dir.exists():
            logger.info(f"✅ llama.cpp already exists: {llama_cpp_path}")
            return str(llama_cpp_dir)
        
        logger.info("📥 Cloning llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_path],
            check=True,
            capture_output=True
        )
        logger.info(f"✅ llama.cpp cloned")
        
        # Compiler
        logger.info("🔨 Building llama.cpp...")
        subprocess.run(
            ["make", "-C", llama_cpp_path],
            check=True,
            capture_output=True
        )
        logger.info(f"✅ llama.cpp compiled")
        
        return str(llama_cpp_dir)
    except Exception as e:
        logger.error(f"❌ Erreur setup llama.cpp: {e}")
        logger.info("Alternative: Install via pip - pip install llama-cpp-python")
        return None

def convert_to_gguf(
    model_path: str,
    output_path: str,
    llama_cpp_path: Optional[str] = None,
    quantize: str = "Q4_K_M"
):
    """Convertir le modèle en format GGUF."""
    try:
        logger.info("=" * 80)
        logger.info("🔄 CONVERSION EN GGUF")
        logger.info("=" * 80)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Option 1: Via llama.cpp
        if llama_cpp_path:
            logger.info(f"📥 Utilisant llama.cpp: {llama_cpp_path}")
            
            # Conversion en GGUF
            convert_script = Path(llama_cpp_path) / "convert.py"
            if not convert_script.exists():
                logger.error(f"❌ Convert script not found: {convert_script}")
                return False
            
            logger.info(f"🔄 Convertir en GGUF...")
            subprocess.run([
                "python3", str(convert_script),
                model_path,
                "--outfile", str(output_file),
                "--outtype", "f16"
            ], check=True)
            logger.info(f"✅ Modèle converti en GGUF")
            
            # Quantizer si demandé
            if quantize:
                logger.info(f"🔄 Quantizer avec {quantize}...")
                quantize_tool = Path(llama_cpp_path) / "quantize"
                if not quantize_tool.exists():
                    logger.warning(f"⚠️  Quantize tool not found, skipping quantization")
                else:
                    output_quant = output_file.with_stem(f"{output_file.stem}-{quantize}")
                    subprocess.run([
                        str(quantize_tool),
                        str(output_file),
                        str(output_quant),
                        quantize
                    ], check=True)
                    logger.info(f"✅ Modèle quantizé: {output_quant}")
                    
                    # Utiliser la version quantizée
                    output_file.unlink()  # Supprimer la version non quantizée
                    output_quant.rename(output_file)
                    logger.info(f"✅ Fichier final: {output_file}")
        else:
            # Option 2: Via llama-cpp-python
            logger.info("📥 Utilisant llama-cpp-python...")
            try:
                from llama_cpp import Llama
                logger.info(f"🔄 Convertir en GGUF via llama-cpp-python...")
                
                # Note: Cette approche dépend de la version
                logger.warning("⚠️  Conversion via llama-cpp-python - utiliser llama.cpp pour meilleur résultat")
                logger.info(f"Veuillez utiliser: llama-cpp-python pour conversion")
                return False
            except ImportError:
                logger.error("❌ llama-cpp-python not installed")
                return False
        
        # Vérifier fichier de sortie
        if output_file.exists():
            file_size = output_file.stat().st_size / 1e9
            logger.info(f"✅ GGUF créé: {output_file} ({file_size:.2f} GB)")
            return True
        else:
            logger.error(f"❌ Output file not created: {output_file}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gguf_model(gguf_path: str):
    """Tester le modèle GGUF."""
    try:
        logger.info("=" * 80)
        logger.info("🧪 TEST MODÈLE GGUF")
        logger.info("=" * 80)
        
        try:
            from llama_cpp import Llama
            
            logger.info(f"📥 Charger modèle GGUF: {gguf_path}")
            llm = Llama(
                model_path=gguf_path,
                n_gpu_layers=-1,  # Utiliser GPU
                n_ctx=2048,
                verbose=False
            )
            logger.info("✅ Modèle chargé")
            
            # Test rapide
            logger.info("🧪 Test génération...")
            prompt = "Bonjour, comment"
            output = llm(prompt, max_tokens=50)
            
            generated_text = output["choices"][0]["text"]
            logger.info(f"✅ Génération réussie:")
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Output: {generated_text}")
            
            return True
        except ImportError:
            logger.warning("⚠️  llama-cpp-python not installed")
            logger.info("Pour tester dans LM Studio: Ouvrir l'interface et charger le modèle GGUF")
            return True  # Pas une erreur, juste un warning
    except Exception as e:
        logger.error(f"❌ Erreur test modèle: {e}")
        return False

def create_lm_studio_readme(gguf_path: str):
    """Créer un README pour LM Studio."""
    try:
        logger.info("📝 Créer README pour LM Studio...")
        
        gguf_file = Path(gguf_path)
        readme_path = gguf_file.parent / "LM_STUDIO_README.md"
        
        content = f"""# 🎯 Modèle Mistral Tiny - Prêt pour LM Studio

## 📋 Informations

- **Modèle:** Mistral Tiny (from scratch)
- **Fichier GGUF:** {gguf_file.name}
- **Taille:** {gguf_file.stat().st_size / 1e9:.2f} GB
- **Quantization:** Q4_K_M (recommandé)
- **Contexte:** 2048 tokens

## 🚀 Utilisation dans LM Studio

1. **Ouvrir LM Studio**
2. **Cliquer sur "Select a model to load"**
3. **Naviguer vers:** {gguf_file.parent}
4. **Sélectionner:** {gguf_file.name}
5. **Cliquer "Load Model"**

## 💡 Paramètres recommandés

- **Temperature:** 0.7
- **Top P:** 0.9
- **Max tokens:** 2048
- **Context:** 2048

## 📚 Documentation

- Mistral: https://www.mistral.ai/
- LM Studio: https://lmstudio.ai/
- llama.cpp: https://github.com/ggerganov/llama.cpp

## ⚙️ Dépannage

**Problème:** Modèle ne charge pas  
**Solution:** Vérifier que le fichier .gguf est valide et accessible

**Problème:** Erreur mémoire  
**Solution:** Réduire "Context Size" ou "Max Tokens" dans LM Studio

## ✅ Modèle prêt!

Le modèle est maintenant utilisable dans LM Studio. Profitez de votre modèle Mistral français!
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ README créé: {readme_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur création README: {e}")
        return False

def main():
    """Pipeline principal d'export GGUF."""
    parser = argparse.ArgumentParser(description="Phase 4 - Export GGUF for LM Studio")
    parser.add_argument("--model-path", type=str, default="./models/mistral_tiny", help="Path to trained model")
    parser.add_argument("--output-path", type=str, default="./models/mistral_tiny.gguf", help="Output GGUF file path")
    parser.add_argument("--llama-cpp-path", type=str, default="./llama.cpp", help="Path to llama.cpp")
    parser.add_argument("--quantize", type=str, default="Q4_K_M", help="Quantization method")
    parser.add_argument("--test", action="store_true", default=True, help="Test the GGUF model")
    parser.add_argument("--setup-llama-cpp", action="store_true", default=False, help="Clone and build llama.cpp")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🔄 PHASE 4 - EXPORT GGUF POUR LM STUDIO")
    logger.info("=" * 80)
    
    # Vérifier dépendances
    if not check_dependencies():
        return False
    
    # Vérifier modèle HuggingFace
    if not ensure_model_is_hf_format(args.model_path):
        logger.error("❌ Model is not in valid HuggingFace format")
        return False
    
    # Setup llama.cpp si demandé
    llama_cpp_path = None
    if args.setup_llama_cpp:
        llama_cpp_path = setup_llama_cpp(args.llama_cpp_path)
        if not llama_cpp_path:
            logger.warning("⚠️  llama.cpp setup failed, but continuing...")
    
    # Convertir en GGUF
    if not convert_to_gguf(
        model_path=args.model_path,
        output_path=args.output_path,
        llama_cpp_path=llama_cpp_path,
        quantize=args.quantize
    ):
        return False
    
    # Tester modèle
    if args.test:
        test_gguf_model(args.output_path)
    
    # Créer README LM Studio
    create_lm_studio_readme(args.output_path)
    
    logger.info("=" * 80)
    logger.info("✅ PHASE 4 COMPLÉTÉE - Modèle prêt pour LM Studio!")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
