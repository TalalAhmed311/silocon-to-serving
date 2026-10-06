terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0" # terraform-aws-eks v21 requires the v6 AWS provider (check the module's versions.tf at v21.26.0)
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "silicon-to-serving", Owner = var.owner, ManagedBy = "terraform", Lab = "p3" }
  }
}
