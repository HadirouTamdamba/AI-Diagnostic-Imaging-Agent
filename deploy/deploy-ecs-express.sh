#!/usr/bin/env bash
#
# One-shot deployment of the Medical Imaging Diagnosis Agent to
# Amazon ECS Express Mode (the AWS-recommended replacement for App Runner).
#
# It provisions, in your AWS account:
#   1. A Secrets Manager secret holding GOOGLE_API_KEY
#   2. An ECR repository + the built Docker image
#   3. The two IAM roles ECS Express Mode needs (created if missing)
#   4. An ECS Express Mode service (Fargate + ALB + auto-scaling + networking)
#
# Prerequisites (see deploy/DEPLOY_AWS.md):
#   - AWS CLI v2 installed and `aws configure` done (credentials + permissions)
#   - Docker running
#   - Your real Gemini key exported:  export GOOGLE_API_KEY=AQ....
#
# Usage:
#   export GOOGLE_API_KEY=AQ...your_key
#   AWS_REGION=eu-west-3 ./deploy/deploy-ecs-express.sh
#
set -euo pipefail

# ----------------------------- Configuration --------------------------------
AWS_REGION="${AWS_REGION:-eu-west-3}"
APP_NAME="${APP_NAME:-medical-imaging-agent}"
ECR_REPO="${ECR_REPO:-$APP_NAME}"
SECRET_NAME="${SECRET_NAME:-$APP_NAME/google-api-key}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_PORT=8501
HEALTH_PATH="/_stcore/health"
CPU="${CPU:-1024}"          # 1 vCPU
MEMORY="${MEMORY:-2048}"    # 2 GB
MIN_TASKS="${MIN_TASKS:-1}"
MAX_TASKS="${MAX_TASKS:-2}"

EXEC_ROLE="${EXEC_ROLE:-ecsTaskExecutionRole}"
INFRA_ROLE="${INFRA_ROLE:-ecsInfrastructureRoleForExpressServices}"

# App configuration passed as (non-secret) environment variables
MODEL_ID="${MODEL_ID:-gemini-flash-latest}"
ENABLE_WEB_SEARCH="${ENABLE_WEB_SEARCH:-true}"
DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-en}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
die() { printf "\n\033[1;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

command -v aws >/dev/null || die "AWS CLI not found. Install it and run 'aws configure'."
command -v docker >/dev/null || die "Docker not found."
docker info >/dev/null 2>&1 || die "Docker daemon is not running."
[ -n "${GOOGLE_API_KEY:-}" ] || die "Export your key first:  export GOOGLE_API_KEY=AQ...  "

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" \
  || die "No valid AWS credentials. Run 'aws configure'."
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

say "Account ${ACCOUNT_ID} | region ${AWS_REGION} | service ${APP_NAME}"

# --------------------------- 1. Secrets Manager -----------------------------
say "Storing GOOGLE_API_KEY in Secrets Manager (${SECRET_NAME})"
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" \
    --secret-string "$GOOGLE_API_KEY" --region "$AWS_REGION" >/dev/null
else
  aws secretsmanager create-secret --name "$SECRET_NAME" \
    --description "Gemini API key for ${APP_NAME}" \
    --secret-string "$GOOGLE_API_KEY" --region "$AWS_REGION" >/dev/null
fi
SECRET_ARN="$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" \
  --region "$AWS_REGION" --query ARN --output text)"

# ------------------------------- 2. ECR image -------------------------------
say "Ensuring ECR repository ${ECR_REPO}"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null

say "Building and pushing image ${IMAGE_URI}"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
# --platform linux/amd64 so the image runs on Fargate even when built on Apple Silicon
docker build --platform linux/amd64 -t "$IMAGE_URI" "$REPO_ROOT"
docker push "$IMAGE_URI"

# ------------------------------- 3. IAM roles -------------------------------
ensure_exec_role() {
  if aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1; then return; fi
  say "Creating ${EXEC_ROLE}"
  aws iam create-role --role-name "$EXEC_ROLE" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
  aws iam attach-role-policy --role-name "$EXEC_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null
}
ensure_exec_role

say "Granting ${EXEC_ROLE} read access to the secret"
aws iam put-role-policy --role-name "$EXEC_ROLE" --policy-name "read-${APP_NAME}-secret" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"${SECRET_ARN}\"}]
  }" >/dev/null

if ! aws iam get-role --role-name "$INFRA_ROLE" >/dev/null 2>&1; then
  cat <<EOF

  The ECS Express Mode infrastructure role '${INFRA_ROLE}' does not exist yet.
  The simplest way to create it (and the execution role) is the one-time ECS
  console first-run for Express Mode, which provisions both roles for you:
    https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html

  Create it, then re-run this script.
EOF
  die "Missing infrastructure role '${INFRA_ROLE}'."
fi

EXEC_ROLE_ARN="$(aws iam get-role --role-name "$EXEC_ROLE" --query Role.Arn --output text)"
INFRA_ROLE_ARN="$(aws iam get-role --role-name "$INFRA_ROLE" --query Role.Arn --output text)"

# --------------------------- 4. Express service -----------------------------
say "Creating ECS Express Mode service ${APP_NAME} (Fargate + ALB + autoscaling)"
PRIMARY_CONTAINER="$(cat <<JSON
{
  "image": "${IMAGE_URI}",
  "containerPort": ${CONTAINER_PORT},
  "environment": [
    {"name": "MODEL_ID", "value": "${MODEL_ID}"},
    {"name": "ENABLE_WEB_SEARCH", "value": "${ENABLE_WEB_SEARCH}"},
    {"name": "DEFAULT_LANGUAGE", "value": "${DEFAULT_LANGUAGE}"},
    {"name": "LOG_LEVEL", "value": "${LOG_LEVEL}"},
    {"name": "LOG_FILE", "value": "/tmp/medical_agent.log"}
  ],
  "secrets": [
    {"name": "GOOGLE_API_KEY", "valueFrom": "${SECRET_ARN}"}
  ]
}
JSON
)"

aws ecs create-express-gateway-service \
  --region "$AWS_REGION" \
  --service-name "$APP_NAME" \
  --execution-role-arn "$EXEC_ROLE_ARN" \
  --infrastructure-role-arn "$INFRA_ROLE_ARN" \
  --primary-container "$PRIMARY_CONTAINER" \
  --health-check-path "$HEALTH_PATH" \
  --cpu "$CPU" --memory "$MEMORY" \
  --scaling-target "{\"minTaskCount\":${MIN_TASKS},\"maxTaskCount\":${MAX_TASKS}}" \
  --monitor-resources

say "Done. Provisioning takes ~3-5 min. Fetch the service URL with:"
cat <<EOF
  aws ecs describe-express-gateway-service --service-name ${APP_NAME} --region ${AWS_REGION} \\
    --query 'service.url' --output text
EOF
