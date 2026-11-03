#!/bin/bash

set -e
set -o pipefail

echo "============================================================"
echo "MOSAIC — Google Cloud Run Deployment"
echo "============================================================"

# CONFIGURATION

PROJECT_ID="mosaic-clinical-trials"
REGION="us-central1"
SERVICE_NAME="mosaic-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

CLOUD_SQL_INSTANCE="${PROJECT_ID}:${REGION}:clinical-trial-db"

SERVICE_ACCOUNT_NAME="mosaic-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_NAME}"
echo ""

# STEP 1: ENABLE REQUIRED GCP APIS

echo "Step 1: Enabling required GCP APIs..."

gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    --project=${PROJECT_ID}

echo "✓ APIs enabled"

# STEP 2: CREATE SERVICE ACCOUNT

echo ""
echo "Step 2: Creating service account..."

gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
    --display-name="MOSAIC API Service Account" \
    --project=${PROJECT_ID} \
    || true

echo "✓ Service account created: ${SERVICE_ACCOUNT_EMAIL}"

# STEP 3: GRANT PERMISSIONS

echo ""
echo "Step 3: Granting service account permissions..."

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

echo "✓ Permissions granted"

# STEP 4: STORE SECRETS IN SECRET MANAGER

echo ""
echo "Step 4: Storing secrets in Secret Manager..."

source .env 2>/dev/null || true

create_or_update_secret() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2

    if gcloud secrets describe ${SECRET_NAME} --project=${PROJECT_ID} &>/dev/null; then

        echo "${SECRET_VALUE}" | gcloud secrets versions add ${SECRET_NAME} \
            --data-file=- \
            --project=${PROJECT_ID}

    else

        echo "${SECRET_VALUE}" | gcloud secrets create ${SECRET_NAME} \
            --data-file=- \
            --replication-policy=automatic \
            --project=${PROJECT_ID}

    fi
}

create_or_update_secret "openai-api-key" "${OPENAI_API_KEY}"
create_or_update_secret "langsmith-api-key" "${LANGSMITH_API_KEY}"
create_or_update_secret "db-password" "${DB_PASSWORD}"

echo "✓ Secrets stored in Secret Manager"

# STEP 5: BUILD DOCKER IMAGE

echo ""
echo "Step 5: Building Docker image for linux/amd64..."

docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_NAME}:latest \
    --file deployment/Dockerfile \
    --push \
    .

echo "✓ Image built and pushed: ${IMAGE_NAME}:latest"

# STEP 6: DEPLOY TO CLOUD RUN

echo ""
echo "Step 6: Deploying to Cloud Run..."

gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --platform=managed \
    --region=${REGION} \
    --service-account=${SERVICE_ACCOUNT_EMAIL} \
    --add-cloudsql-instances=${CLOUD_SQL_INSTANCE} \
    --set-env-vars="\
GCP_PROJECT_ID=${PROJECT_ID},\
GCP_REGION=${REGION},\
GCS_BUCKET_NAME=${GCS_BUCKET_NAME},\
DB_HOST=/cloudsql/${CLOUD_SQL_INSTANCE},\
DB_PORT=5432,\
DB_NAME=${DB_NAME},\
DB_USER=${DB_USER},\
OPENAI_EMBEDDING_MODEL=${OPENAI_EMBEDDING_MODEL},\
OPENAI_CHAT_MODEL=${OPENAI_CHAT_MODEL},\
LANGSMITH_PROJECT=${LANGSMITH_PROJECT},\
LANGSMITH_TRACING_V2=${LANGSMITH_TRACING_V2},\
API_HOST=0.0.0.0,\
API_PORT=8000,\
API_ENV=production" \
    --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
LANGSMITH_API_KEY=langsmith-api-key:latest,\
DB_PASSWORD=db-password:latest" \
    --min-instances=0 \
    --max-instances=3 \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --concurrency=1 \
    --no-allow-unauthenticated \
    --project=${PROJECT_ID}

echo "✓ Deployed to Cloud Run"

# STEP 7: GET SERVICE URL

echo ""
echo "Step 7: Getting service URL..."

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo "✓ Service URL: ${SERVICE_URL}"

# STEP 8: TEST DEPLOYMENT

echo ""
echo "Step 8: Testing deployment with health check..."

sleep 10

HEALTH_RESPONSE=$(curl -s \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    "${SERVICE_URL}/api/v1/health")

echo "Health check response:"
echo ${HEALTH_RESPONSE} | python3 -m json.tool

# DEPLOYMENT COMPLETE

echo ""
echo "============================================================"
echo "MOSAIC deployment complete"
echo "============================================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test your deployment:"
echo "curl -s \\"
echo "  -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"task\": \"Find completed trials with missing results\"}' \\"
echo "  ${SERVICE_URL}/api/v1/analyze | python3 -m json.tool"
echo ""
echo "View logs in GCP Console:"
echo "https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/logs?project=${PROJECT_ID}"
echo ""
echo "Stop Cloud SQL when done to save cost:"
echo "gcloud sql instances patch clinical-trial-db --activation-policy=NEVER --project=${PROJECT_ID}"
echo "============================================================"