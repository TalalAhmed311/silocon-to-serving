variable "region" {
  type    = string
  default = "us-east-1"
}
variable "owner" { type = string }
variable "cluster_name" {
  type        = string
  description = "The EKS cluster from infra/aws/eks (terraform output cluster_name there)"
}
variable "checkpoint_retention_days" {
  type    = number
  default = 14
}
