terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0" # UNVERIFIED range: pin to the current major from the provider changelog and commit .terraform.lock.hcl
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  default_labels = {
    project    = "silicon-to-serving"
    owner      = var.owner
    managed-by = "terraform"
    lab        = "p3-9"
  }
}
