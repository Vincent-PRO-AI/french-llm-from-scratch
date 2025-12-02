#!/bin/bash
# Script de déploiement Azure Container Apps pour French LLM API
# Usage: ./deploy_azure.sh [resource-group] [location]

set -e

# Configuration
RG="${1:-french-llm-rg}"
LOC="${2:-westeurope}"
APP_ENV="french-llm-env"
APP_NAME="french-llm-api"
IMAGE_NAME="french-llm-api:latest"
ACR_NAME="frenchllmcr$RANDOM"

echo "=========================================="
echo "🚀 Déploiement French LLM API sur Azure"
echo "=========================================="
echo ""
echo "Resource Group: $RG"
echo "Location: $LOC"
echo "Container Registry: $ACR_NAME"
echo ""

# Vérifier Azure CLI
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI n'est pas installé."
    echo "Installez-le depuis: https://learn.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé."
    exit 1
fi

# Login Azure
echo "🔐 Connexion à Azure..."
az account show > /dev/null 2>&1 || az login

# Créer le Resource Group
echo "📦 Création du Resource Group..."
az group create --name "$RG" --location "$LOC" --output none
echo "✅ Resource Group créé: $RG"

# Créer l'Azure Container Registry
echo "🐳 Création du registre de conteneurs..."
az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RG" \
    --sku Basic \
    --admin-enabled true \
    --output none
echo "✅ Container Registry créé: $ACR_NAME"

# Login au registry
echo "🔑 Connexion au registre..."
az acr login --name "$ACR_NAME"

# Récupérer le login server
ACR_LOGIN=$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query loginServer -o tsv)
echo "📍 Registry URL: $ACR_LOGIN"

# Build l'image Docker
echo "🏗️  Build de l'image Docker..."
docker build -t "$ACR_LOGIN/$IMAGE_NAME" -f deploy/fastapi/Dockerfile .
echo "✅ Image construite"

# Push l'image
echo "📤 Push de l'image vers ACR..."
docker push "$ACR_LOGIN/$IMAGE_NAME"
echo "✅ Image uploadée"

# Récupérer les credentials ACR
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query passwords[0].value -o tsv)

# Créer l'environnement Container Apps
echo "🌍 Création de l'environnement Container Apps..."
az containerapp env create \
    --name "$APP_ENV" \
    --resource-group "$RG" \
    --location "$LOC" \
    --output none
echo "✅ Environnement créé"

# Déployer l'application
echo "🚢 Déploiement de l'application..."
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --environment "$APP_ENV" \
    --image "$ACR_LOGIN/$IMAGE_NAME" \
    --registry-server "$ACR_LOGIN" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8000 \
    --ingress external \
    --cpu 2 \
    --memory 4Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars MODEL_ID=vincent-pro-ai/french-llm-from-scratch \
    --output none
echo "✅ Application déployée"

# Récupérer l'URL
FQDN=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT TERMINÉ"
echo "=========================================="
echo ""
echo "🌐 URL de l'API: https://$FQDN"
echo "📚 Documentation: https://$FQDN/docs"
echo "💬 Interface Chat: https://$FQDN/"
echo ""
echo "📋 Commandes utiles:"
echo "  - Logs: az containerapp logs show -n $APP_NAME -g $RG --follow"
echo "  - Restart: az containerapp revision restart -n $APP_NAME -g $RG"
echo "  - Supprimer: az group delete -n $RG --yes"
echo ""
echo "🧪 Test:"
echo "  curl -X POST https://$FQDN/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\":\"Bonjour\",\"max_new_tokens\":50}'"
echo ""
