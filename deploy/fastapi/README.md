# French LLM from scratch – API d'inférence (FastAPI)

Cette API expose le modèle `vincent-pro-ai/french-llm-from-scratch` via `/generate`.

## Lancer en local (sans Docker)

```bash
# Option A: utiliser le venv du projet
source .venv/bin/activate
pip install -r deploy/fastapi/requirements.txt

# Utiliser le modèle local pour éviter un téléchargement
export MODEL_LOCAL_DIR="trained_models/huggingface/french-llm-from-scratch"
uvicorn deploy.fastapi.app:app --host 0.0.0.0 --port 8000
```

Tester:
```bash
curl -s http://localhost:8000/healthz | jq
curl -s -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explique brièvement la différence entre apprentissage supervisé et non supervisé."}' | jq -r .text
```

## Docker (local)

```bash
# Construire
docker build -t french-llm-api:cpu -f deploy/fastapi/Dockerfile .

# Lancer (téléchargera le modèle depuis HF au premier run)
docker run --rm -p 8000:8000 \
  -e MODEL_ID=vincent-pro-ai/french-llm-from-scratch \
  french-llm-api:cpu
```

## Déploiement Azure Container Apps (CPU)

Pré-requis: Azure CLI, abonnement Azure.

```bash
# 1) Variables
RG=french-llm-rg
LOC=westeurope
APP_ENV=french-llm-env
APP_NAME=french-llm-api
IMAGE=french-llm-api:cpu
REG=llmcr$RANDOM

# 2) Build local et push vers ACR (registre géré Azure)
az group create -n $RG -l $LOC
az acr create -n $REG -g $RG --sku Basic
az acr login -n $REG
ACR_LOGIN=$(az acr show -n $REG -g $RG --query loginServer -o tsv)

docker build -t $ACR_LOGIN/$IMAGE -f deploy/fastapi/Dockerfile .
docker push $ACR_LOGIN/$IMAGE

# 3) Créer l'environnement Container Apps
az containerapp env create -g $RG -n $APP_ENV -l $LOC

# 4) Déployer l'app
az containerapp create -g $RG -n $APP_NAME \
  --environment $APP_ENV \
  --image $ACR_LOGIN/$IMAGE \
  --ingress external --target-port 8000 \
  --cpu 2 --memory 4Gi \
  --env-vars MODEL_ID=vincent-pro-ai/french-llm-from-scratch

# 5) Récupérer l'URL
az containerapp show -g $RG -n $APP_NAME --query properties.configuration.ingress.fqdn -o tsv
```

Appels HTTP:
```bash
ENDPOINT=https://<fqdn-container-app>

curl -s $ENDPOINT/healthz | jq
curl -s -X POST $ENDPOINT/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explique brièvement la différence entre apprentissage supervisé et non supervisé."}' | jq -r .text
```

## Alternatives prêtes à l'emploi

- Hugging Face Inference Endpoints (PaaS): interface web → choisir le repo → déployer.
- TGI (Text Generation Inference) GPU:
  ```bash
  docker run --gpus all -p 8080:80 \
    -e MODEL_ID=vincent-pro-ai/french-llm-from-scratch \
    ghcr.io/huggingface/text-generation-inference:2.4
  # API Swagger sur http://localhost:8080/docs
  ```
- llama.cpp server (CPU): bon pour GGUF FP32, latence réduite avec petits modèles.

Notes:
- Le modèle (260M) tourne confortablement sur CPU (2 vCPU / 4–8 Go RAM). Pour GPU, adapter l'image (PyTorch CUDA) et la taille de VM.
- GGUF quantisé n'est pas recommandé ici (vocab=32001). Utiliser Transformers (ce service) pour un fonctionnement garanti.