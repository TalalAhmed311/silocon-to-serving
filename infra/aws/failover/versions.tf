terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# Route 53 is global; CloudWatch alarms live in each serving region.
provider "aws" {
  region = var.primary_region
  default_tags { tags = local.tags }
}
provider "aws" {
  alias  = "primary"
  region = var.primary_region
  default_tags { tags = local.tags }
}
provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
  default_tags { tags = local.tags }
}

locals {
  tags = { Project = "silicon-to-serving", Owner = var.owner, ManagedBy = "terraform", Lab = "c1" }
}
