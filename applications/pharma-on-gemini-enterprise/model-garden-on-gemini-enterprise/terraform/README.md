# Deploy Model Garden Agent with Terraform

This folder contains the declarative Terraform configuration to package the `model-garden-on-gemini-enterprise` agent into a secure, non-root custom Docker container and deploy it to Agent Platform (Agent Runtime).

## Features
* **Custom Container Build (`image_spec`):** Builds a non-root Docker container automatically from the source code archive using the `Dockerfile` inside your project.
* **Agent Identity Integration:** Assigns the unique, least-privilege Agent Identity principal (`principal://agents...`) for accessing Agent Platform and Telemetry services by default.
* **Standard Service Account Fallback:** Gracefully supports deploying the container under a standard custom Service Account if your GCP project is a standalone sandbox (not part of a Google Workspace or Cloud Identity organization).
* **Cost Optimized Scaling:** Defaults `min_instances` to `0` to allow the container to scale to zero when idle.
* **Observability & Telemetry:** Wires up OpenTelemetry variables directly to Google Cloud Observability (metrics, logs, and traces) using `roles/telemetry.writer`.

---

## Prerequisites
1. **Terraform v1.0+** installed locally.
2. **Google Cloud Provider 7.28.0+** (this configuration requires `version = ">= 7.28.0"` to support `image_spec`).
3. Standard GCP APIs enabled in your target project:
   * `aiplatform.googleapis.com` (Agent Platform)
   * `cloudbuild.googleapis.com` (for building the custom image)
   * `artifactregistry.googleapis.com`
4. Suitable GCP credentials configured via:
   ```bash
   gcloud auth application-default login
   ```

---

## Configuration Variables

Configure these variables in a `terraform.tfvars` file or pass them via the command line (`-var`).

| Variable | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`project_id`** | `string` | **(Required)** The Google Cloud Project ID. | *None* |
| **`region`** | `string` | The GCP region to deploy the Reasoning Engine to. | `us-central1` |
| **`display_name`** | `string` | User-facing name of the Reasoning Engine. | `Model Garden Agent` |
| **`model_name`** | `string` | Default LLM name (e.g. Claude). | `claude-opus-4-7` |
| **`model_location`** | `string` | Region where the Model Garden endpoint resides. | `global` |
| **`log_level`** | `string` | Container logging level (`DEBUG`, `INFO`, `WARN`). | `INFO` |
| **`enable_telemetry`** | `bool` | Enable Cloud Observability tracing and auto-logging. | `true` |
| **`capture_message_content`**| `string` | Capture LLM inputs/outputs (`EVENT_ONLY`, `true`, `false`).| `EVENT_ONLY` |
| **`min_instances`** | `number` | Min container instances (set to `0` for scale-to-zero). | `0` |
| **`max_instances`** | `number` | Max instances during heavy scaling. | `5` |
| **`use_agent_identity`** | `bool` | True to use secure Agent Identity (requires project in an org).| `true` |
| **`custom_service_account`**| `string` | Fallback SA email if `use_agent_identity = false`. | `null` |

---

## Deployment Guide

### 1. Initialize Terraform
Navigate to this directory and run initialization to download providers and set up modules:
```bash
terraform init
```

### 2. Validate syntax
Ensure the files are syntactically correct and compliant:
```bash
terraform validate
```

### 3. Plan
Perform a dry-run to review the changes Terraform will make to your GCP project:
```bash
terraform plan -var="project_id=YOUR_PROJECT_ID" -var="region=us-central1"
```

### 4. Deploy (Apply)
Apply the changes to package, build, and deploy the agent:
```bash
terraform apply -var="project_id=YOUR_PROJECT_ID" -var="region=us-central1"
```

On success, the output will display the generated Reasoning Engine Resource ID and the actual runtime effective identity principal.

---

## Org-dependency and Fallback (Agent Identity vs Service Account)
* **If your project is inside a Cloud Identity / Workspace Organization:** Leave `use_agent_identity = true` (default). Terraform retrieves the exact federated identity dynamically using the resource's `effective_identity` output attribute, and grants the required `roles/aiplatform.user` and `roles/telemetry.writer` permissions directly to that principal, maximizing security and eliminating manual construction errors.
* **If your project is standalone:** Set `use_agent_identity = false` and provide a pre-created Service Account email via `custom_service_account`. The Reasoning Engine container will run under that Service Account, and Terraform will grant permissions to it instead.
