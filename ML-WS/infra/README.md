# ML-WS Deployment to GCP Cloud Run

This directory contains Terraform configurations to deploy the ML-WS service to Google Cloud Run.

## Prerequisites

- A Google Cloud Platform (GCP) account.
- A GCP project with billing enabled.
- A service account with the following roles:
  - `roles/serviceusage.serviceUsageAdmin`
  - `roles/cloudbuild.builds.builder`
  - `roles/run.admin`
  - `roles/storage.admin`
  - `roles/iam.serviceAccountUser`
- A service account key (JSON) for authentication.
- A GitHub repository for your project.

## Manual GCP Setup

1. **Create a GCP Project**:

   - Go to the [GCP Console](https://console.cloud.google.com/).
   - Create a new project or select an existing one.
   - Note the project ID.

2. **Enable Required APIs**:

   - Go to [APIs & Services > Library](https://console.cloud.google.com/apis/library)
   - Enable the following APIs:
     - Cloud Resource Manager API (`cloudresourcemanager.googleapis.com`)
     - Cloud Run API (`run.googleapis.com`)
     - Cloud Build API (`cloudbuild.googleapis.com`)
     - Artifact Registry API (`artifactregistry.googleapis.com`)

3. **Create a Service Account**:

   - Go to IAM & Admin > Service Accounts.
   - Create a new service account with the roles listed above.
   - Create a key (JSON) for this service account and download it.

4. **Create a GCS Bucket for Terraform State**:

   - Go to Cloud Storage.
   - Create a new bucket (e.g., `your-terraform-state-bucket`).
   - This bucket will store the Terraform state.

5. **Set Up GitHub Connection**:
   - Go to [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
   - Click "Connect Repository"
   - Select "GitHub"
   - Follow the steps to connect your GitHub repository
   - Note the connection name (it will look like `projects/YOUR_PROJECT_ID/locations/global/connections/github-connection`)
   - Update the `connection` field in `main.tf` with this connection name

## Terraform Configuration

1. **Update `terraform.tfvars`**:

   - Set `gcp_project_id` to your GCP project ID.
   - Optionally, change `gcp_region` if needed.

2. **Update `main.tf`**:
   - Set the `bucket` in the `backend "gcs"` block to your GCS bucket name.
   - Update the GitHub owner and repo name in the `google_cloudbuild_trigger` resource.
   - Add the GitHub connection name from step 5 above.

## GitHub Actions Setup

1. **Add Secrets to GitHub**:

   - Go to your GitHub repository > Settings > Secrets.
   - Add the following secrets:
     - `GCP_PROJECT_ID`: Your GCP project ID.
     - `GCP_SA_KEY`: The content of your service account key JSON file.

2. **Push Changes**:
   - Push your changes to the `main` branch.
   - GitHub Actions will automatically deploy the service to Cloud Run.

## Deployment

- The GitHub Actions workflow will:
  - Build and push the Docker image to GCR.
  - Deploy the image to Cloud Run.
  - Output the Cloud Run service URL.

## Switching GCP Accounts

To switch GCP accounts:

1. Update the `GCP_SA_KEY` secret in GitHub with the new service account key.
2. Update the `gcp_project_id` in `terraform.tfvars`.
3. Push the changes to GitHub.
4. GitHub Actions will automatically deploy to the new account and output the new URL.

## Troubleshooting

- Ensure the service account has the necessary permissions.
- Check the GitHub Actions logs for any errors.
- Verify the GCP APIs are enabled in your project.
- Make sure the GitHub connection is properly set up in Cloud Build.
- If you get a "Request contains an invalid argument" error, check that the GitHub connection name in `main.tf` matches the one in Cloud Build.

TIMEBOX:
1. the cloud run is deploying on commit
2. have no idea where the endpoint url is stored.
3. should sign into personal aws and try to redeploy EB but using the new web service
4. main issue is that it needs access to actually hitting the endpoint
