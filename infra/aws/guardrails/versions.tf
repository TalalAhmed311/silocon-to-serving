terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.70" # loose on purpose: used by roots pinned to provider v5 (single-node) and v6 (eks)
    }
  }
}
