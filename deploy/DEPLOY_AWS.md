# Deploying to AWS

Two paths are provided:

- **`deploy/terraform/`** — Infrastructure-as-Code (ECS Fargate + ALB + Secrets Manager +
  IAM + CloudWatch). **Costs nothing until `terraform apply`** — ideal to showcase the
  cloud architecture in a portfolio without any billing. See
  [deploy/terraform/README.md](terraform/README.md).
- **ECS Express Mode** (below) — a one-command managed deploy (AWS's App Runner
  replacement) for an actual live URL.

> **Zero-cost portfolio tip:** keep the Terraform module as reviewable IaC (don't apply),
> or apply it briefly, screenshot the running app, then `terraform destroy` (a few cents).
> Either way the *code* demonstrates the AWS competency.

---

## Why not App Runner?

AWS **App Runner is closed to new customers** (announced 2026). Only accounts that
already used App Runner can create services. AWS's recommended, App Runner-equivalent
replacement is **Amazon ECS Express Mode**: one command provisions a full stack
(Fargate + Application Load Balancer + auto-scaling + networking) and gives you an
HTTPS URL. There is **no extra charge for Express Mode** — you pay only for the
underlying resources (Fargate task + ALB).

> If your AWS account *is* an existing App Runner customer and you prefer App Runner,
> the only difference is the final step: replace `create-express-gateway-service` with
> an `apprunner create-service` call pointing at the same ECR image and secret. Ask and
> we'll swap it.

---

## Prerequisites

1. **AWS CLI v2** installed and configured:
   ```bash
   # macOS
   brew install awscli
   aws configure   # access key, secret, default region eu-west-3
   ```
   The IAM user/role needs permissions for ECR, ECS, IAM, and Secrets Manager.
2. **Docker** running (to build the image).
3. Your **Gemini API key** (`AQ...` or `AIza...`).
4. The two IAM roles ECS Express Mode requires. Easiest: do the one-time
   [ECS Express Mode console first-run](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html),
   which auto-creates `ecsTaskExecutionRole` and `ecsInfrastructureRoleForExpressServices`.
   (The script below creates the execution role if missing, but the infrastructure role
   must exist first.)

Region used throughout: **eu-west-3 (Paris)** — App Runner *and* ECS Express Mode are
available there, keeping data and latency in France.

---

## Option A — One-shot deploy (first deployment)

```bash
export GOOGLE_API_KEY=AQ...your_real_key      # never commit this
AWS_REGION=eu-west-3 ./deploy/deploy-ecs-express.sh
```

The script:
1. Stores the key in **Secrets Manager** (`medical-imaging-agent/google-api-key`).
2. Creates the **ECR** repo, builds the image `--platform linux/amd64`, pushes it.
3. Ensures the **execution role** can read the secret.
4. Creates the **ECS Express Mode** service (port 8501, health `/_stcore/health`,
   1 vCPU / 2 GB, autoscale 1→2 tasks).

Provisioning takes ~3–5 min. Get the URL:
```bash
aws ecs describe-express-gateway-service --service-name medical-imaging-agent \
  --region eu-west-3 --query 'service.url' --output text
```

Tunable via env vars: `APP_NAME`, `CPU`, `MEMORY`, `MIN_TASKS`, `MAX_TASKS`,
`MODEL_ID`, `ENABLE_WEB_SEARCH`, `DEFAULT_LANGUAGE`.

---

## Option B — Continuous deployment (GitHub Actions)

After the first deploy exists, [`.github/workflows/deploy-ecs.yml`](../.github/workflows/deploy-ecs.yml)
rebuilds and redeploys the image. It is `workflow_dispatch` (manual) by default;
uncomment the `push` trigger for auto-deploy on every push to `main`.

One-time setup:
1. Create a **GitHub OIDC provider** in AWS and an IAM role `github-actions-ecs-role`
   trusting your repo, with ECR + ECS Express permissions
   ([guide](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws)).
2. Add repository **Variables** (Settings → Secrets and variables → Actions → Variables):
   `AWS_REGION=eu-west-3`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY=medical-imaging-agent`,
   `ECS_SERVICE=medical-imaging-agent`.

---

## Secret handling

- **Option A (script)** injects `GOOGLE_API_KEY` from **Secrets Manager** into the task
  (`secrets` → `valueFrom`), so the key never appears in the task definition env or the repo.
- The CI action updates the **image**; keep the secret managed by Secrets Manager (set once
  by the script). If a redeploy ever drops the secret binding, re-run the script, or add
  the key as a GitHub secret passed through `environment-variables` (less secure — visible
  in the task def).

Rotate the key anytime:
```bash
aws secretsmanager put-secret-value --secret-id medical-imaging-agent/google-api-key \
  --secret-string AQ...new_key --region eu-west-3
```

---

## Cost (rough, eu-west-3)

- **Fargate** 1 vCPU / 2 GB, 1 task always on: ~**$35–45 / month**.
- **ALB**: ~**$18 / month** + traffic.
- **ECR / Secrets Manager**: cents.
- Scale `MIN_TASKS=0` is not supported for an always-on web app; to avoid idle cost for a
  demo, tear down when unused (below).

Gemini API is billed separately by Google (free tier: ~20 requests/day on `gemini-flash-latest`;
enable billing in Google AI Studio for production).

---

## Teardown

```bash
aws ecs delete-express-gateway-service --service-name medical-imaging-agent --region eu-west-3
aws ecr delete-repository --repository-name medical-imaging-agent --force --region eu-west-3
aws secretsmanager delete-secret --secret-id medical-imaging-agent/google-api-key \
  --recovery-window-in-days 7 --region eu-west-3
```
