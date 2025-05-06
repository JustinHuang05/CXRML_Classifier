terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  backend "gcs" {
    bucket = "terraform-state-bucket-cxrml3"
    prefix = "ml-ws"
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Enable required GCP APIs
resource "google_project_service" "run" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "build" {
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "registry" {
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# Build and push Docker image using Cloud Build
resource "google_cloudbuild_trigger" "build_trigger" {
  name        = "ml-ws-build"
  description = "Build ML-WS Docker image"
  filename    = "cloudbuild.yaml"
  included_files = ["ML-WS/**"]
  github {
    owner = "JustinHuang05"
    name  = "CXRML_Classifier"
    push {
      branch = "^main$"
    }
  }
}

# Deploy to Cloud Run
resource "google_cloud_run_service" "ml_ws" {
  name     = "ml-ws"
  location = var.gcp_region
  template {
    spec {
      containers {
        image = "gcr.io/${var.gcp_project_id}/ml-ws:latest"
        ports {
          container_port = 8080
        }
      }
    }
  }
  traffic {
    percent         = 100
    latest_revision = true
  }
  depends_on = [google_cloudbuild_trigger.build_trigger]
}

# Output the Cloud Run service URL
output "service_url" {
  value = google_cloud_run_service.ml_ws.status[0].url
} 