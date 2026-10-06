terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70" # pin a minor range; run `terraform init -upgrade` deliberately, never by accident
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "silicon-to-serving"
      Owner     = var.owner
      ManagedBy = "terraform"
      Lab       = var.lab
    }
  }
}
