variable "region" {
  type    = string
  default = "us-east-1"
}
variable "owner" { type = string }
variable "budget_email" { type = string }
variable "monthly_budget_usd" {
  type    = number
  default = 300
}
variable "kubernetes_version" {
  type    = string
  default = "1.33" # pick a version EKS currently supports in standard support (EKS docs); UNVERIFIED default
}
variable "gpu_instance_types" {
  type    = list(string)
  default = ["g6.xlarge"]
}
variable "gpu_capacity_type" {
  type    = string
  default = "ON_DEMAND" # or "SPOT" (P3.6 builds an on-demand + spot pair)
}
variable "gpu_min" {
  type    = number
  default = 0 # scale-to-zero: you pay for GPUs only while pods need them
}
variable "gpu_max" {
  type    = number
  default = 2
}
variable "allowed_api_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to reach the public EKS API endpoint (your IP /32). Never the whole internet."
}
