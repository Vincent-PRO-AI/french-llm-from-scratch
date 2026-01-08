# 📤 Guide de Publication sur Hugging Face

Guide complet pour publier le modèle **french-llm-from-scratch** (200k steps) sur Hugging Face Hub.

---

## 📋 **PRÉREQUIS**

### 1. Compte Hugging Face
- Créer un compte sur [huggingface.co](https://huggingface.co/join)
- Générer un token d'accès : [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Type de token : **Write** (pour upload)

### 2. Packages Python
```bash
pip install huggingface-hub transformers safetensors tokenizers
```

### 3. Connexion
```bash
# Option 1 : CLI (recommandé)
hf auth login

# Option 2 : Variable d'environnement
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

---

## 🚀 **PROCÉDURE COMPLÈTE**

### **Étape 1 : Export du modèle** (5-10 min)

Convertit le checkpoint PyTorch en format Hugging Face (safetensors + config.json).

```bash
cd /home/vincent/code/defendGPT

python3 scripts/export_to_huggingface.py \
  --checkpoint trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt \
  --output-dir trained_models/huggingface/french-llm-from-scratch
```

**Ce qui est créé** :
- ✅ `model.safetensors` (~1 GB) - Poids du modèle
- ✅ `config.json` - Configuration architecture GPT-2
- ✅ `README.md` - Model card complète

**Vérifications** :
- [ ] model.safetensors créé sans erreur
- [ ] config.json contient vocab_size=32000, n_layer=18, n_head=16
- [ ] README.md lisible et complet

---

### **Étape 2 : Préparation du tokenizer** (2-5 min)

Convertit le tokenizer Fineweb-32k custom en format Hugging Face.

```bash
python3 scripts/prepare_tokenizer_for_hf.py \
  --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --output-dir trained_models/huggingface/french-llm-from-scratch
```

**Ce qui est créé** :
- ✅ `tokenizer.json` - Définition BPE
- ✅ `tokenizer_config.json` - Config Hugging Face
- ✅ `special_tokens_map.json` - Mapping tokens spéciaux

**Vérifications** :
- [ ] Test d'encodage/décodage réussi
- [ ] Vocab size = 32000
- [ ] Tokens spéciaux définis (bos, eos, pad, unk)

---

### **Étape 3 : Test local** (2-3 min)

Vérifie que le modèle charge correctement avant l'upload.

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger depuis le dossier local
model_dir = 'trained_models/huggingface/french-llm-from-scratch'
print(f'📂 Chargement depuis {model_dir}...')

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

print(f'✅ Tokenizer chargé: {len(tokenizer)} tokens')
print(f'✅ Modèle chargé: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

# Test génération
prompt = 'Bonjour, comment vas-tu ?'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'\n🧪 Test génération:')
print(f'Prompt: {prompt}')
print(f'Résultat: {result}')
"
```

**Vérifications** :
- [ ] Chargement sans erreur
- [ ] Génération produit du texte cohérent
- [ ] Nombre de paramètres correct (~260M)

---

### **Étape 4 : Upload vers Hugging Face** (10-20 min selon connexion)

Upload le modèle complet sur le Hub.

```bash
# Choisir le nom du repo (format: username/model-name)
REPO_ID="vincent-pro-ai/french-llm-from-scratch"

python3 scripts/upload_to_huggingface.py \
  --local-dir trained_models/huggingface/french-llm-from-scratch \
  --repo-id "$REPO_ID"
```

**Options** :
- `--private` : Créer un repo privé (sinon public)
- `--token` : Fournir token manuellement (sinon utilise huggingface-cli login)

**Progression** :
```
📋 Vérification des fichiers...
   ✅ model.safetensors (1024.5 MB)
   ✅ config.json (0.0 MB)
   ✅ README.md (0.0 MB)
   ✅ tokenizer.json (2.1 MB)
   ✅ tokenizer_config.json (0.0 MB)

📦 Création/vérification du repo: vincent-pro-ai/french-llm-from-scratch
✅ Repo créé/existant: https://huggingface.co/vincent-pro-ai/french-llm-from-scratch

⬆️  Upload des fichiers...
Uploading: 100%|████████████████████| 1.03G/1.03G [03:42<00:00, 4.61MB/s]
✅ Upload terminé!
```

**Vérifications** :
- [ ] URL du modèle accessible : `https://huggingface.co/{REPO_ID}`
- [ ] README.md affiché correctement
- [ ] Onglet "Files" montre tous les fichiers

---

### **Étape 5 : Test depuis le Hub** (2-3 min)

Vérifie que le modèle est téléchargeable et utilisable.

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

# Remplacer par votre REPO_ID
REPO_ID = 'vincent-pro-ai/french-llm-from-scratch'

print(f'📥 Téléchargement depuis Hugging Face Hub...')
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForCausalLM.from_pretrained(REPO_ID)

print(f'✅ Modèle téléchargé et chargé!')

# Test génération
prompt = 'Expliquez-moi l\'intelligence artificielle'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8, top_k=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'\n🧪 Génération:')
print(result)
"
```

---

## 📊 **MÉTRIQUES À DOCUMENTER**

Avant publication, récupérer les métriques finales :

```bash
# Loss finale
tail -1 trained_models/runs/french_medium_mega_finetune_200k/metrics.jsonl

# Perplexité
python3 -c "
import json
from pathlib import Path
import math

metrics_file = Path('trained_models/runs/french_medium_mega_finetune_200k/metrics.jsonl')
with open(metrics_file) as f:
    lines = [json.loads(line) for line in f]

# Dernière métrique avec val_loss
final_metric = next((m for m in reversed(lines) if m.get('val_loss')), None)

if final_metric:
    val_loss = final_metric['val_loss']
    perplexity = math.exp(val_loss)
    print(f'Step: {final_metric[\"step\"]}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Perplexity: {perplexity:.2f}')
"

# Échantillons de génération
tail -20 trained_models/runs/french_medium_mega_finetune_200k/samples.txt
```

**À ajouter dans README.md** (section Performance) :
- Final validation loss
- Perplexity
- 2-3 exemples de génération de qualité

---

## 🔄 **ALTERNATIVES : OLLAMA**

Si tu veux aussi publier sur Ollama (pour usage local) :

### 1. Créer un Modelfile

```bash
cat > trained_models/huggingface/Modelfile << 'EOF'
FROM ./model.safetensors

PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

TEMPLATE """{{ .Prompt }}"""

SYSTEM """Tu es un assistant IA français. Réponds de manière claire et utile."""
EOF
```

### 2. Créer le modèle Ollama

```bash
cd trained_models/huggingface/french-llm-from-scratch

# Convertir en GGUF (requis pour Ollama)
# Note: Nécessite llama.cpp ou conversion manuelle
ollama create french-llm-from-scratch -f Modelfile
```

### 3. Tester

```bash
ollama run french-llm-from-scratch "Bonjour, parle-moi de l'IA"
```

---

## 📝 **CHECKLIST FINALE**

Avant de publier publiquement :

### Qualité du modèle
- [ ] Loss < 5.0 sur validation
- [ ] Génération cohérente en français
- [ ] Pas de répétitions excessives
- [ ] Grammaire correcte

### Documentation
- [ ] README.md complet avec exemples
- [ ] Licence spécifiée (MIT recommandé)
- [ ] Dataset sources citées
- [ ] Limitations documentées

### Technique
- [ ] model.safetensors valide
- [ ] config.json correct (vocab_size, architecture)
- [ ] tokenizer fonctionnel
- [ ] Test de chargement réussi

### Éthique
- [ ] Pas de contenu offensant/illégal
- [ ] Biais potentiels documentés
- [ ] Usage approprié clarifié

---

## 🎯 **COMMANDES RAPIDES (TL;DR)**

```bash
# 1. Export modèle
python3 scripts/export_to_huggingface.py

# 2. Préparer tokenizer
python3 scripts/prepare_tokenizer_for_hf.py

# 3. Test local
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('trained_models/huggingface/french-llm-from-scratch'); print('✅ OK')"

# 4. Connexion HF
hf auth login

# 5. Upload
python3 scripts/upload_to_huggingface.py --repo-id "vincent-pro-ai/french-llm-from-scratch"
```

---

## 🐛 **DÉPANNAGE**

### Erreur : "No module named 'apex'"
```bash
# Ignorer, apex est optionnel pour l'export
# Le script fonctionne sans
```

### Erreur : "Invalid token"
```bash
# Reconnecter à HF
hf auth logout
hf auth login
```

### Erreur : "File too large"
```bash
# Utiliser git-lfs pour les gros fichiers (automatique avec huggingface-cli)
git lfs install
```

### Tokenizer ne fonctionne pas
```bash
# Vérifier que tokenizer.json est valide
python3 -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('trained_models/tokenizers/fineweb-32k/tokenizer.json'); print(f'✅ Vocab: {t.get_vocab_size()}')"
```

---

## 📚 **RESSOURCES**

- [Hugging Face Hub Docs](https://huggingface.co/docs/hub/index)
- [Model Card Guide](https://huggingface.co/docs/hub/model-cards)
- [Safetensors Format](https://huggingface.co/docs/safetensors/index)
- [Tokenizers Library](https://huggingface.co/docs/tokenizers/index)

---

**Durée totale estimée** : 20-40 minutes (selon vitesse réseau)

🚀 **Prêt à publier !**
